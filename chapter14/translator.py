import os
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------#
# 1. BPE Tokenization (SentencePiece)
# ---------------------#
# 先运行以下命令训练 BPE 模型（只需运行一次）：
# spm.SentencePieceTrainer.Train('--input="data\\en2cn\\train_en.txt" --model_prefix=en_bpe --vocab_size=16000 --model_type=bpe --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')
# spm.SentencePieceTrainer.Train('--input="data\\en2cn\\train_zh.txt" --model_prefix=zh_bpe --vocab_size=16000 --model_type=bpe --character_coverage=0.9995 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')

sp_en = spm.SentencePieceProcessor()
sp_en.load('en_bpe.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('zh_bpe.model')


def tokenize_en(text):
    return sp_en.encode(text, out_type=int)


def tokenize_cn(text):
    return sp_cn.encode(text, out_type=int)

# 中文和英文一致,取英文。
PAD_ID = sp_en.pad_id()  # 1
UNK_ID = sp_en.unk_id()  # 0
BOS_ID = sp_en.bos_id()  # 2
EOS_ID = sp_en.eos_id()  # 3


# ---------------------#
# 2. Dataset & DataLoader
# ---------------------#
class TranslationDataset(Dataset):
    ## 初始化方法，读取英文和中文训练文本。然后给每个句子前后增加<bos>和<eos>。 为了防止训练时显存不足，对于长度超过限制的
    ## 句子进行过滤。
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer, max_len=100):
        with open(src_file, encoding='utf-8') as f:
            src_lines = f.read().splitlines()
        with open(trg_file, encoding='utf-8') as f:
            trg_lines = f.read().splitlines()
        assert len(src_lines) == len(trg_lines)
        self.pairs = []
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        for src, trg in zip(src_lines, trg_lines):
            # 每个句子前边增加<bos>后边增加<eos>
            src_ids = [BOS_ID] + self.src_tokenizer(src) + [EOS_ID]
            trg_ids = [BOS_ID] + self.trg_tokenizer(trg) + [EOS_ID]
            # 只保留输入和输出序列token数同时小于max_len的训练样本。
            if len(src_ids) <= max_len and len(trg_ids) <= max_len:
                self.pairs.append((src_ids, trg_ids))  # <-- 直接保存token id序列

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.pairs[idx]
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)

    ## 对一个batch的输入和输出token序列，依照最长的序列长度，用<pad> token进行填充，确保一个batch的数据形状一致，组成一个tensor。
    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        trg_lens = [len(x) for x in trg_batch]
        src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_ID)
        trg_pad = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_ID)
        return src_pad, trg_pad, src_lens, trg_lens


# ---------------------#
# 3. Model Definitions with Attention
# ---------------------#
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2 + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [1, batch, hid_dim] -> [batch, 1, hid_dim]
        # encoder_outputs: [src_len, batch, hid_dim*2] -> [batch, src_len, hid_dim*2]
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(1, src_len, 1)  # [batch, src_len, hid_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)  # [batch, src_len]


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=2):
        super().__init__()
        self.n_layers=n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.bi_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True)
        self.fc_hidden = nn.ModuleList([nn.Linear(hid_dim * 2, hid_dim) for _ in range(n_layers)])
        self.fc_cell = nn.ModuleList([nn.Linear(hid_dim * 2, hid_dim) for _ in range(n_layers)])

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        outputs, (hidden, cell) = self.bi_lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)  # [src_len, batch, hid_dim*2]

        # 重塑隐藏状态和细胞状态: [n_layers * 2, batch, hid_dim] -> [n_layers, 2, batch, hid_dim]
        hidden = hidden.view(self.n_layers, 2, -1, hidden.size(2))
        cell = cell.view(self.n_layers, 2, -1, cell.size(2))

        # 为每一层处理前向和后向状态
        final_hidden = []
        final_cell = []

        # concat the final forward and backward hidden state and pass through a linear layer

        for layer in range(self.n_layers):
            h_cat = torch.cat((hidden[layer][-2], hidden[layer][-1]), dim=1)
            c_cat = torch.cat((cell[layer][-2], cell[layer][-1]), dim=1)
            h_layer = torch.tanh(self.fc_hidden[layer](h_cat)).unsqueeze(0)
            c_layer = torch.tanh(self.fc_cell[layer](c_cat)).unsqueeze(0)

            final_hidden.append(h_layer)
            final_cell.append(c_layer)

        hidden_concat = torch.cat(final_hidden, dim=0)
        cell_concat = torch.cat(final_cell, dim=0)
        return outputs, hidden_concat, cell_concat


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention, n_layers=2):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(hid_dim * 2 + emb_dim, hid_dim,num_layers=n_layers)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, output_dim)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        # input_token: [batch]
        input_token = input_token.unsqueeze(0)  # [1, batch]
        embedded = self.embedding(input_token)  # [1, batch, emb_dim]

        last_hidden = hidden[-1].unsqueeze(0)
        a = self.attention(last_hidden, encoder_outputs, mask)  # [batch, src_len]
        a = a.unsqueeze(1)  # [batch, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, src_len, enc_hid_dim*2]
        weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, enc_hid_dim*2]
        weighted = weighted.permute(1, 0, 2)  # [1, batch, enc_hid_dim*2]

        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch, emb_dim + enc_hid_dim*2]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [1, batch, hid_dim]

        output = output.squeeze(0)  # [batch, hid_dim]
        embedded = embedded.squeeze(0)  # [batch, emb_dim]
        weighted = weighted.squeeze(0)  # [batch, enc_hid_dim*2]

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # [batch, output_dim]

        return prediction, hidden, cell, a.squeeze(1)  # attention weights for visualization


def create_mask(src, src_len):
    # src: [src_len, batch]
    mask = (src != PAD_ID).permute(1, 0)  # [batch, src_len]
    return mask


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_len)

        input_token = trg[0]
        mask = create_mask(src, src_len)

        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[t] = output
            input_token = trg[t]

        return outputs


# ---------------------#
# 4. Training
# ---------------------#
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    step_loss = 0  # 用于累计每个step的loss
    step_count = 0  # 当前step计数器

    for i, (src, trg, src_len, _) in enumerate(iterator):
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # 更新损失统计
        step_loss += loss.item()
        epoch_loss += loss.item()
        step_count += 1

        # 每100个step打印一次
        if (i + 1) % 100 == 0:
            avg_step_loss = step_loss / step_count
            print(f'Step [{i + 1}/{len(iterator)}] | Loss: {avg_step_loss:.4f}')
            step_loss = 0  # 重置step损失
            step_count = 0  # 重置step计数器

    # 打印最后一批数据（如果不足100）
    if step_count > 0:
        avg_step_loss = step_loss / step_count
        print(f'Step [{len(iterator)}/{len(iterator)}] | Loss: {avg_step_loss:.4f}')

    return epoch_loss / len(iterator)  # 返回整个epoch的平均loss


# ---------------------#
# 5. Main
# ---------------------#
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TranslationDataset('data\\en2cn\\train_en.txt', 'data\\en2cn\\train_zh.txt', tokenize_en, tokenize_cn)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=TranslationDataset.collate_fn)

    INPUT_DIM = sp_en.get_piece_size()
    OUTPUT_DIM = sp_cn.get_piece_size()
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512

    attention = Attention(HID_DIM).to(device)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM).to(device)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attention).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    N_EPOCHS = 1
    CLIP = 1

    for epoch in range(N_EPOCHS):
        loss = train(model, loader, optimizer, criterion, CLIP)
        print(f'Epoch {epoch + 1}/{N_EPOCHS} | Loss: {loss:.4f}')
        torch.save(model.state_dict(), 'seq2seq_bpe_attention.pt')

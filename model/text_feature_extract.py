from torch import nn
import torch

class TextExtract_gru(nn.Module):

    def __init__(self, opt):
        """
        GRU Network
        Consideration inspired by https://github.com/xx-adeline/MFPE/blob/main/src/model/text_feature_extract.py#L6
        """
        super(TextExtract_gru, self).__init__()

        self.opt = opt
        self.last_lstm = opt.last_lstm  # Consider renaming this variable to 'last_gru'
        self.embedding = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        # Replace LSTM with GRU
        self.gru = nn.GRU(512, 1024, num_layers=1, bidirectional=True, bias=False)

    def forward(self, caption_id, text_length):

        text_embedding = self.embedding(caption_id)
        text_embedding = self.dropout(text_embedding)
        # Call the GRU processing function
        feature = self.calculate_different_length_gru(text_embedding, text_length, self.gru)

        return feature

    def calculate_different_length_gru(self, text_embedding, text_length, gru):

        text_length = text_length.view(-1)
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :]
        sort_text_length = text_length[sort_index]
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(sortlength_text_embedding,
                                                                  sort_text_length.cpu(),
                                                                  batch_first=True)

        packed_feature, hn = gru(packed_text_embedding)  # Only hn for GRU
        total_length = text_embedding.size(1)
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True, total_length=total_length) # Including [feature, length]

        if self.last_lstm:  # Consider renaming this variable to 'last_gru'
            hn = torch.cat([hn[0, :, :], hn[1, :, :]], dim=1)[unsort_index, :]
            return hn
        else:
            unsort_feature = sort_feature[0][unsort_index, :]
            unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]+ unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2
            return unsort_feature.permute(0, 2, 1).contiguous().unsqueeze(3)

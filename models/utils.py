
import torch



def create_padding_mask(seq, pad_idx):
	"""Creating tensor for masking pad tokens of scaled dot-product logits
	Args:
		seq: Tensor with shape (batch_size, seq_len)
			 [ [3, 20, 17, 0 ,0 ] [...] [...] ]
		pad_idx: idx to be padded.
	Return:
		seq: Tensor with shape (batch_size, 1, 1, , seq_len)
			 [ [ [ [0, 0, 0, 1, 1] [...] [...] ] ] ]
	"""
	seq = torch.eq(seq, pad_idx).double()
	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, None, None, :]


def create_look_ahead_mask(seq_len):
	"""Creating Tensor used for future token masking
	Args:
		src_len: scales
	Returns:
		mask: Tensor with shape (seq_len, seq_len)
	"""
	mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
	return mask


def create_transformer_masks(src, tgt, pad_idx):
	"""Creating three masks for TransformerEncoder and -Decoder 
	Args:
		src: shape (batch_size, src_len)
		tgt: shape (batch_size, tgt_len)
	Return:
		enc_pad_mask: masking pad tokens in the encoder
		combined_mask: used to pad and mask future tokens a.k.a `future_mask`
		dec_pad_mask: masking the encoder outputs in 2nd attention block
	"""
	# Encoder padding mask
	enc_pad_mask = create_padding_mask(src, pad_idx)

	# Used in the 2nd attention block in the decoder.
	# This padding mask is used to mask the encoder outputs.
	dec_pad_mask = create_padding_mask(src, pad_idx)

	# Used in the 1st attention block in the decoder.
	# It is used to pad and mask future tokens in the input received by
	# the decoder.
	look_ahead_mask = create_look_ahead_mask(tgt.shape[1])
	dec_target_padding_mask = create_padding_mask(tgt, pad_idx)
	combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)

	return enc_pad_mask, combined_mask, dec_pad_mask



if __name__ == "__main__":
	x = torch.tensor([[7, 6, 1, 0, 0], [1, 2, 3, 0, 0], [2, 0, 0, 0, 0]])
	y = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 0, 0], [2, 0, 0, 0, 0]])	
	print(x)
	print(y)
	m = create_padding_mask(x, 0)


	a,b,c = create_transformer_masks(x,y,0)
	print(a)
	print(b)
	print(c)

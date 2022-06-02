from metrics import ArcMarginProduct

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import RobertaPreTrainedModel, RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, SequenceClassifierOutput, _CONFIG_FOR_DOC, RobertaClassificationHead

from transformers.file_utils import (
	add_code_sample_docstrings,
	add_end_docstrings,
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	replace_return_docstrings,
)

class RobertaForSentenceEmbedding(RobertaPreTrainedModel):
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.config = config

		self.roberta = RobertaModel(config, add_pooling_layer=False)
		self.classifier = ArcMarginProduct(in_features = self.roberta.encoder.layer[-1].output.dense.out_features, out_features = self.num_labels, s=30, m=0.5)

		# Initialize weights and apply final processing
		self.post_init()

	# @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	# @add_code_sample_docstrings(
	#     processor_class=_TOKENIZER_FOR_DOC,
	#     checkpoint="cardiffnlp/twitter-roberta-base-emotion",
	#     output_type=SequenceClassifierOutput,
	#     config_class=_CONFIG_FOR_DOC,
	#     expected_output="'optimism'",
	#     expected_loss=0.08,
	# )
	def forward(
		self,
		input_ids = None,
		attention_mask = None,
		token_type_ids = None,
		position_ids = None,
		head_mask = None,
		inputs_embeds = None,
		labels = None,
		output_attentions = None,
		output_hidden_states = None,
		return_dict = None,
		extract = False,
	):
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.roberta(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		token_embeddings = outputs[0] #First element of model_output contains all token embeddings
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

		if extract:
			return sentence_embeddings
		logits = self.classifier(sentence_embeddings, labels)

		loss = None
		if labels is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(logits, labels)

		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

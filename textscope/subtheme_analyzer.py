import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from .config import SUBTHEMES
import nltk
nltk.download('punkt_tab')


class SubthemeAnalyzer:
    def __init__(self) -> None:
        self.subthemes = SUBTHEMES
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.task = 'Given a set of words forming a topic, determine whether the text discusses the topic'

    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\\Topic: {query}'

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed_batch(self, texts: list, batch_size: int = 32) -> Tensor:
        """Embed a list of texts in batches, returning normalized embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_dict = self.tokenizer(
                batch, max_length=512, padding=True, truncation=True, return_tensors='pt',
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def _flatten_keywords(self, subthemes: list) -> tuple:
        """Flatten subthemes into (keyword_texts, keyword_to_subtheme_index) mapping."""
        keyword_texts = []
        keyword_to_subtheme = []
        is_nested = any(isinstance(item, list) for item in subthemes)

        if is_nested:
            for idx, theme in enumerate(subthemes):
                for kw in theme:
                    keyword_texts.append(
                        self._get_detailed_instruct(self.task, kw)
                    )
                    keyword_to_subtheme.append(idx)
        else:
            for idx, theme in enumerate(subthemes):
                keyword_texts.append(
                    self._get_detailed_instruct(self.task, theme)
                )
                keyword_to_subtheme.append(idx)

        return keyword_texts, keyword_to_subtheme

    def __main_analysis(self, theme: str, sent: str) -> float:
        instruct = [self._get_detailed_instruct(self.task, theme)]
        doc = [sent]
        input_texts = instruct + doc
        batch_dict = self.tokenizer(
            input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt',
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        return scores.tolist()[0][0]

    def is_nested_subtheme(self, lst):
        return any(isinstance(item, list) for item in lst)

    def analyze(self, text: str, profile: str) -> list:
        """Batched: embed all sentences and keywords once, return scores per subtheme."""
        if not text:
            return []

        if profile not in SUBTHEMES:
            raise ValueError(f"Profile '{profile}' not found in the subthemes configuration.")
        subthemes = SUBTHEMES[profile]
        n_subthemes = len(subthemes)

        sentences = sent_tokenize(text)
        if not sentences:
            return [0.0] * n_subthemes

        sent_embeddings = self._embed_batch(sentences)
        keyword_texts, keyword_to_subtheme = self._flatten_keywords(subthemes)
        kw_embeddings = self._embed_batch(keyword_texts)

        sim_matrix = (kw_embeddings @ sent_embeddings.T) * 100

        subthemes_scores = []
        for subtheme_idx in range(n_subthemes):
            kw_indices = [i for i, s in enumerate(keyword_to_subtheme) if s == subtheme_idx]
            max_sim = sim_matrix[kw_indices].max().item()
            subthemes_scores.append(max_sim)

        return subthemes_scores

    def analyze_bin(self, text: str, profile: str, thr: float = 85.) -> list:
        """Batched: embed all sentences and keywords once, return binary presence per subtheme."""
        if not text:
            return []

        if profile not in SUBTHEMES:
            raise ValueError(f"Profile '{profile}' not found in the subthemes configuration.")
        subthemes = SUBTHEMES[profile]
        n_subthemes = len(subthemes)

        sentences = sent_tokenize(text)
        if not sentences:
            return [0] * n_subthemes

        sent_embeddings = self._embed_batch(sentences)
        keyword_texts, keyword_to_subtheme = self._flatten_keywords(subthemes)
        kw_embeddings = self._embed_batch(keyword_texts)

        sim_matrix = (kw_embeddings @ sent_embeddings.T) * 100

        subtheme_pres = []
        for subtheme_idx in range(n_subthemes):
            kw_indices = [i for i, s in enumerate(keyword_to_subtheme) if s == subtheme_idx]
            max_sim = sim_matrix[kw_indices].max().item()
            subtheme_pres.append(1 if max_sim > thr else 0)

        return subtheme_pres

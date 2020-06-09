import random


class CitationMaker:
    def __init__(self, processor):
        self.nlp = processor

    def make_newline_and_label_by_option(self, c, citation_config: dict):
        """
        make newline by option
        :param c: str | citation string
        :param citation_config: dict | e.g. {
            "min_citations_per_output_file": 50,
            "use_citation_index": True,
            "newline_option": "dist",
            "break_ratio": 0.45,
            "nlp_processor": "spacy"
        }
        :return: str | modified citation
        """
        doc = None
        tokens = []
        labels = []

        nlp_processor = citation_config.get("nlp_processor", None)
        intra_citation_newline_type = citation_config.get(
            "intra_citation_newline_type", None
        )
        dist_ratio = citation_config.get("dist_ratio", 0.45)

        if nlp_processor == "spacy" or intra_citation_newline_type == "spacy":
            doc = self.nlp(c)
        else:
            doc = c.split(" ")

        i = 0
        if intra_citation_newline_type == "spacy":
            for sentence in doc.sents:
                for word in sentence:
                    tokens.append(word.text)
                    if i == 0:
                        labels.append("B-CIT")
                    else:
                        labels.append("I-CIT")
                    i = i + 1
                tokens.append("\n")
                labels.append("I-CIT")
                i = i + 1
        elif intra_citation_newline_type == "dist":
            dist = 0
            for word in doc:
                if nlp_processor == "spacy":
                    tokens.append(word.text)
                else:
                    tokens.append(word)
                dist += 1
                th = dist / float(len(doc))
                if th >= float(dist_ratio):
                    tokens.append("\n")
                    if i == 0:
                        labels.append("B-CIT")
                    else:
                        labels.append("I-CIT")
                    i = i + 1
                    dist = 0
                if i == 0:
                    labels.append("B-CIT")
                else:
                    labels.append("I-CIT")
                i = i + 1
        elif intra_citation_newline_type == "random":
            for word in doc:
                if nlp_processor == "spacy":
                    tokens.append(word.text)
                else:
                    tokens.append(word)
                if i == random.randint(0, len(doc)):
                    tokens.append("\n")
                    if i == 0:
                        labels.append("B-CIT")
                    else:
                        labels.append("I-CIT")
                    i = i + 1
                if i == 0:
                    labels.append("B-CIT")
                else:
                    labels.append("I-CIT")
                i = i + 1
        else:
            for word in doc:
                if nlp_processor == "spacy":
                    tokens.append(word.text)
                else:
                    tokens.append(word)
                if i == 0:
                    labels.append("B-CIT")
                else:
                    labels.append("I-CIT")
                i = i + 1
        return tokens, labels

    @staticmethod
    def get_styled_indexed_citation(c: str, i: int, style=0):
        """
        metthod to get formatted citation index
        :param c: str | citation string
        :param i: int | citation index
        :param style: int | greater than 0, e.g. i., (i), [i], i)
        :return: str, int | modified citation , style
        """
        if style == 0:
            c = "[" + str(i) + "]" + " " + c
        elif style == 1:
            c = str(i) + "." + " " + c
        elif style == 2:
            c = "(" + str(i) + ")" + " " + c
        elif style == 3:
            c = str(chr(96 + i)) + ")" + " " + c
        else:
            c = str(i) + ")" + " " + c
        return c, style

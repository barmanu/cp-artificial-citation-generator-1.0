import random

from .citation import CitationMaker


class Generator:
    """
    given a list of citation, it generates the artificial citation sections with modified citation (e.g. adding newlines, index)
    """

    def __init__(self, other_ref_itation_list, nlp_processor):
        self.nlp = nlp_processor
        self.citations = other_ref_itation_list
        self.cm = CitationMaker(processor=self.nlp, )

    def get_citations_generator(self, config: dict):
        """
        generate modified citation section
        :param config: dict | {"min_citations_per_output_file": 50, "use_citation_index": True, "newline_option": "spacy/dist/random/ignore"}
        :return: list | list of lists contains citations
        """
        min_citations_per_output_file = min(
            config["min_citations_per_output_file"], len(self.citations)
        )
        citation_index_style = -1
        citation_collection = []
        citation_index = 1

        for i, c in enumerate(self.citations):
            uci = config.get("use_citation_index", True)
            if uci:
                if citation_index_style == -1:
                    citation_index_style = random.randint(0, 5)
                (
                    c,
                    citation_index_style,
                ) = CitationMaker.get_styled_indexed_citation(
                    c=c, i=citation_index, style=citation_index_style
                )
            citation_index = citation_index + 1
            tokens, labels = self.cm.make_newline_and_label_by_option(
                c=c, citation_config=config
            )
            if citation_index_style == 3 and len(citation_collection) == 26:
                yield citation_collection
                citation_collection = []
                citation_index_style = random.randint(0, 5)
                citation_index = 1
            elif len(
                    citation_collection
            ) > min_citations_per_output_file + random.randint(
                0, int(min_citations_per_output_file)
            ):
                yield citation_collection
                citation_collection = []
                citation_index_style = random.randint(0, 5)
                citation_index = 1
            else:
                citation_collection.append((tokens, labels))
        if len(citation_collection) > 0:
            yield citation_collection

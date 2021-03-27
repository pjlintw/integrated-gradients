"""Dataset loading script for WikiTableQuestions sparated by tab."""
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "WikiTableQuestions"

class WikiTableQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for OntoNotes 4.0"""

    def __init__(self, **kwargs):
        """BuilderConfig for OntoNotes 4.0.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiTableQuestionsConfig, self).__init__(**kwargs)


class WikiTableQuestions(datasets.GeneratorBasedBuilder):
    """OntoNotes 4.0."""
    
    BUILDER_CONFIGS = [
        WikiTableQuestionsConfig(name='WikiTableQuestions', description="WikiTableQuestions dataset.")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "length": datasets.Value("int32"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "label": datasets.features.ClassLabel(
                                names=[
                                "who",
                                "what",
                                "when",   
                                "where",
                                "why",
                                "how",
                                "which",
                                "whose",

                            ]
                        )
                }
            )
        )

    def _split_generators(self, dl_manager):
        
        train_file = "./data/wikiTable/sample.train"
        dev_file = "./data/wikiTable/sample.dev"
        test_file ="./data/wikiTable/sample.test"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_file}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_file}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_file}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            
            for line in f:
                line_list = line.strip().split("\t")
                label, length, sentence = line_list

                if len(line_list) != 3:
                    print(line_list)
                yield guid, {
                    "id": str(guid),
                    "tokens": sentence.split(),
                    "length": int(length),
                    "label": label,
                }



if __name__ == "__main__":
    WikiTableQuestions(__name__)






























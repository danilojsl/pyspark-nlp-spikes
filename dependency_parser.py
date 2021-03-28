import sparknlp
from pyspark.sql.types import *
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.sql.functions import udf

"""
Configuration:
spark-nlp 2.7.5
pyspark 2.4.7
spark 2.4.7
java 8
python 3.6.9
"""


def get_heads_children_info(dependency_metadata):
    root_index = ['-1']
    heads = root_index + [dm['head'] for dm in dependency_metadata]
    children = []
    for index, _ in enumerate(heads):
        if index != 0:
            children.append([i for i, e in enumerate(heads) if e == str(index)])

    return heads[1:], children


if __name__ == '__main__':

    spark = sparknlp.start()

    test_ds = spark.createDataFrame(
        ["So what happened?",
         "It should continue to be defanged.",
         "That too was stopped."],
        StringType()).toDF("text")

    test_ds.show(5, False)

    main_path = "/home/dburbano/IdeaProjects/JSL/spark-nlp/src/test/resources/"
    conllU_training_file = main_path + "parser/labeled/train_small.conllu.txt"

    pos_tagger = PerceptronModel.pretrained()

    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    tokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")

    dependency_parser = DependencyParserApproach() \
        .setInputCols(["sentence", "pos", "token"]) \
        .setOutputCol("dependency") \
        .setConllU(conllU_training_file) \
        .setNumberOfIterations(10)

    pipeline_dependency_parser = Pipeline().setStages([
        document_assembler,
        sentence_detector,
        tokenizer,
        pos_tagger,
        dependency_parser
    ])

    dp_df = pipeline_dependency_parser.fit(test_ds).transform(test_ds)
    dp_df.printSchema()
    dp_df.select(dp_df["dependency.metadata"]).show(5, False)

    schema = StructType([
        StructField("heads", ArrayType(StringType()), False),
        StructField("children", ArrayType(StringType()), False)
    ])

    getHeadsAndChildrenUDF = udf(lambda z: get_heads_children_info(z), schema)

    metadata_df = dp_df \
        .withColumn("metadata", getHeadsAndChildrenUDF(dp_df["dependency.metadata"]))

    metadata_df.withColumn("heads", metadata_df["metadata.heads"]) \
        .withColumn("children", metadata_df["metadata.children"]) \
        .select("heads", "children")\
        .show(5, False)

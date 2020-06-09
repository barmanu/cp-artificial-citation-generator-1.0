import argparse
import json
import os
import random
import sys
from datetime import datetime
from zipfile import ZipFile as MZ

import boto3
import git
import mlflow
import mlflow.keras
import pandas as pd
import spacy
from tqdm import tqdm

from generator.data_generator.citation_section_generator import Generator


def validation_for_data(df: pd.DataFrame, target_tag: str):
    """
    validation for data
    :param df: pd.DataFrame | input DataFrame
    :param target_tag: str: target_tag
    :return: bool | True
    """
    b = False
    cols = list(df.columns)
    if "text" in cols and "label" in cols:
        if target_tag in df.label.tolist():
            b = True
    return b


def load_json(path):
    """
    :param path: str | path to json
    :return: dict
    """
    d = None
    with open(path) as f:
        d = json.load(f)
    f.close()
    d = dict(zip(d.values(), d.keys()))
    return d


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--output-dir",
        dest="output",
        default="./nlp/exps/output",
        help="output dir",
    )
    parser.add_argument(
        "--training-mode", dest="train", default=False, help="training"
    )
    parser.add_argument(
        "--data-config",
        dest="data_config_path",
        default="./config/data_config.json",
        help="data generation config",
    )
    parser.add_argument(
        "--feature-config",
        dest="feature_config_path",
        default="./config/feature_config.json",
        help="pre-processing/feature engg config",
    )
    parser.add_argument(
        "--model-config",
        dest="model_config_path",
        default="./config/model_config.json",
        help="training model config",
    )
    parser.add_argument(
        "--mlflow-server-url",
        dest="mlflow_server",
        default="https://mlflow.caps.dev.dp.elsevier.systems",
        help="mlflow server path",
    )
    parser.add_argument(
        "--store-at-server",
        dest="store_at_server",
        default=True,
        help="to store data and artifacts or not",
    )

    args = parser.parse_args()

    # init: name, output-dir, timestamp
    repo = git.Repo(search_parent_directories=True)
    exp_name = "REPO:" + repo.remote("origin").url
    mlflow.set_tracking_uri(args.mlflow_server)
    mlflow.set_experiment(exp_name)

    timestmp = str(datetime.now()).replace(" ", "~")

    args.output = os.path.join(args.output, timestmp)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # data gen config loaded
    data_gen_config = None
    with open(args.data_config_path) as jf:
        data_gen_config = json.load(jf)
    jf.close()
    print("### Data Gen. Options given:\n")
    for k, v in data_gen_config.items():
        print(f"--{k} {v}")
    print("\n")

    # downloading data and validate
    df = None
    s3_client = boto3.client("s3")
    if not os.path.exists(data_gen_config["data_df"]):
        print("### Downloading Dataset")
        filename = "other_ref_citations_from_64k_data_set.csv"
        s3_client.download_file(
            "manuscript64k",
            "other_refs/" + filename,
            data_gen_config["data_df"],
        )
        print("### Downloading Complete")
    else:
        df = pd.read_csv(data_gen_config["data_df"])
    if not validation_for_data(
            df=df, target_tag=data_gen_config["target_tag"]
    ):  # validating data
        print("### Error: Wrong Corpus !!!")
        sys.exit(0)
    print("### Dataset Validated")

    # sorting according to length
    df["len"] = df.apply(lambda x: len(x.text.split(" ")), axis=1)
    other_refs = df[df["label"] == data_gen_config["target_tag"]][
        "text"
    ].tolist()
    lim = int(data_gen_config["max_citations_from_corpus"])
    if lim < 0.0:
        lim = len(other_refs)

    # data gen. init
    nlp_processor = spacy.load("en_core_web_sm")
    print("### Spacy Loaded ...")
    gen = Generator(
        other_ref_itation_list=other_refs[: lim - 1],
        nlp_processor=nlp_processor,
    )
    citation_section_generator = gen.get_citations_generator(
        config=data_gen_config
    )

    data_gen_stats = []

    # data gen.
    print("### Data Generation Started ...")
    max_len = -1
    for fake_file in tqdm(citation_section_generator):
        gold_words = []
        gold_tags = []
        citation_count = 0
        token_count = 0
        file_intra_citation_newline = 0
        file_inter_citation_newline = 0

        for pair in fake_file:
            citation_tokens, citation_labels = pair
            file_intra_citation_newline += citation_tokens.count("\n")
            if data_gen_config["inter_citation_newline_type"] == "consistent":
                for i in range(int(data_gen_config["num_of_inter_newlines"])):
                    citation_tokens.append("\n")
                    token_count += len(citation_tokens) + 1
                    citation_labels.append("I-CIT")
                file_inter_citation_newline += int(
                    data_gen_config["num_of_inter_newlines"]
                )
            elif data_gen_config["inter_citation_newline_type"] == "random":
                k = random.randint(
                    0, int(data_gen_config["num_of_inter_newlines"])
                )
                for i in range(k):
                    citation_tokens.append("\n")
                    token_count += len(citation_tokens) + 1
                    citation_labels.append("I-CIT")
                file_inter_citation_newline += k

            for w in citation_tokens:
                gold_words.append(w)
            for t in citation_labels:
                gold_tags.append(t)
            citation_count = citation_count + 1

        file_to_write = "data-" + str(datetime.now()).replace(" ", "~") + ".csv"
        df = pd.DataFrame({"x": gold_words, "y": gold_tags})
        df.to_csv(os.path.join(args.output, file_to_write))
        if len(df) > max_len:
            max_len = len(df)
        temp = {
            "name": file_to_write,
            "citations": citation_count,
            "tokens": token_count,
            "intra_citation_newlines": file_intra_citation_newline,
            "inter_citation_newline": file_inter_citation_newline,
        }
        data_gen_stats.append(temp)

    # logging data gen params
    for k, v in data_gen_config.items():
        mlflow.log_param(k, v)

    with open(os.path.join(args.output, "data-gen-config.json"), "w") as jf:
        json.dump(data_gen_config, jf)
    jf.close()

    data_gen_stats = pd.DataFrame(data_gen_stats)
    data_gen_stats.to_csv(os.path.join(args.output, "data_generation_stats.csv"))

    zip_name = "citation-bio-labelled-data-" + timestmp + ".zip"
    zip_path = os.path.join(args.output, zip_name)
    with MZ(zip_path, "w") as zips:
        for file in os.listdir(args.output):
            if not file.endswith(".zip"):
                zips.write(os.path.join(args.output, file))

    if args.store_at_server:
        mlflow.log_artifact(zip_path)

    # saving the data in S3
    print("saving data")
    os.system('aws s3 sync ' + args.output + ' s3://manuscript64k/other_refs/artificial_data/' + timestmp)
    print("### Data Generation Finished ...")
    print("### Done !!!")


if __name__ == "__main__":
    main()

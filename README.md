# Creative Commons License annotation of CommonCrawl

> *Raw CommonCrawl crawls, annotated with potential Creative Commons license information*

The licensing information is extracted from the web pages based on whether they link to Creative Commons licenses but false positives may occur! While further filtering based on the location type of the license should improve the precision (e.g. by removing hyperlink (a_tag) references), false positives may still occur. **Note that no quality filter occurs to ensure a wide coverage!** However, a column is added to indicate whether a sample exists in the FineWeb-2 dataset.

By default, we only process the following languages, although you can change this by add a `languages` key to your YAML config file with the languages that you want. Default:

```yaml
languages:
- afr_Latn
- deu_Latn
- eng_Latn
- fra_Latn
- fry_Latn
- ita_Latn
- nld_Latn
- spa_Latn
```

The following fields are extracted:

In some cases, multiple licenses are found on a single page. All licenses are collected in `potential_licenses`. From these, the "best guess" is selected
based on three criteria. All licenses are sorted based on these properties and the first one is detected. E.g. licenses in the meta tag are preferred over an `a` tag, or (location equal) a license in the `head` or footer has preference over one in the body.

1. location_preference_order: meta_tag, json-ld, link_tag, a_tag
2. head_preference_order: True, False
3. footer_preference_order: True, False

Based on these criteria, the "best guessed" license is picked as the one in the `license_*` columns. Potential disagreement between multiple licenses is given in `license_disagreement`.

- text: the extracted text (unmodified)
- id: WARC-Record-ID
- dump: Common Crawl crawl
- url: original url for document
- date: crawl date
- file_path: file path on the S3 bucket
- license_abbr: the license type. Possible values: "cc-unknown" (recommended to filter this one out), "by", "by-sa", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd", "zero", "certification", "mark". If multiple licenses were found, `potential_licenses` contains all of them.
- license_version: the license version, e.g. "4.0"
- license_location: the location where the license was found. Possible values: "meta_tag", "json-ld", "link_tag", "a_tag"
- license_in_head: whether the license was found inside a `head` HTML element
- license_in_footer: whether the license was found inside a `footer` HTML element, or an HTML element that had `footer` in the ID or class name
- potential_licenses:
  - abbr: list of all found license abbreviations
  - version: list of all found license versions
  - location: list of all found license locations
  - in_head: list of whether licenses were found in the head
  - in_footer: list of whether licenses were found in a footer
- license_parse_error: whether there was a problem when trying to extract the license, e.g. an unparseable HTML document
- license_disagreement: whether the `potential_licenses["abbr"]` disagree, i.e., different types of licenses were found. License *versions* are not included in the comparison!
- language_script: the script of the language as detected by `glotlid`
- language: the language, as detected by `glotlid`
- language_score: the language identification confidence score
- found_in_fw2: whether this sample was found in FineWeb-2 based on the. Crawls that are more recent than FW2 (everything after 2024-18) are marked as None. English crawl is also marked as None!

## Installation

Simply pip install this repository. E.g., for an editable install:

```shell
python -m pip install -e .
```

## Usage

While `local` alternatives are given for running the pipeline on your local machine, the recommended use is via SLURM through `scripts/run_slurm.py`. Usage is facilitated via the SLURM launch scripts in `slurm/launch.slurm`. To use the scripts, you do need to take care of some things:

1. The pipeline includes a check to see whether a sample exists in the FineWeb-2 dataset as a quality signal. Download the DuckDB files of the languages that you are interested in. By default we process the languages mentioned above, so to download those to the expected `duckdbs/fineweb-2` directory inside this project root:

```shell
huggingface-cli download BramVanroy/fineweb-2-duckdbs fw2-afr_Latn.duckdb --local-dir duckdbs/fineweb-2/ --repo-type dataset
huggingface-cli download BramVanroy/fineweb-2-duckdbs fw2-deu_Latn.duckdb --local-dir duckdbs/fineweb-2/ --repo-type dataset
huggingface-cli download BramVanroy/fineweb-2-duckdbs fw2-fra_Latn.duckdb --local-dir duckdbs/fineweb-2/ --repo-type dataset
huggingface-cli download BramVanroy/fineweb-2-duckdbs fw2-fry_Latn.duckdb --local-dir duckdbs/fineweb-2/ --repo-type dataset
huggingface-cli download BramVanroy/fineweb-2-duckdbs fw2-ita_Latn.duckdb --local-dir duckdbs/fineweb-2/ --repo-type dataset
huggingface-cli download BramVanroy/fineweb-2-duckdbs fw2-nld_Latn.duckdb --local-dir duckdbs/fineweb-2/ --repo-type dataset
huggingface-cli download BramVanroy/fineweb-2-duckdbs fw2-spa_Latn.duckdb --local-dir duckdbs/fineweb-2/ --repo-type dataset
```

2. In the SLURM scripts under `slurm/`, change the constants/variables in capital letters with your specific use-case (account, partition, etc.).
3. Update the config under `configs/` depending on your hardware. This may take some trial and error on your specific system configuration but the default values are expected to work. In the `configs/config-slurm.yaml` make sure to updated the root dir where you saved the DuckDB files

Now you can submit the job to start processing a specific crawl, e.g.

```bash
sbatch launch.slurm CC-MAIN-2024-51
```

Output of the first step will be saved, by default, in `output-main/` and the final data (added column whether the sample exists in FineWeb-2) in `output/`.

## Progress

See [the dataset page](https://huggingface.co/datasets/BramVanroy/CommonCrawl-CreativeCommons).


## Citation

If you use this library or data that was created with it, please cite:

```bibtex
@software{Vanroy_CommonCrawl-CreativeCommons_2025,
  author = {Vanroy, Bram},
  license = {GPL-3.0},
  month = feb,
  title = {{CommonCrawl-CreativeCommons}},
  url = {https://github.com/BramVanroy/CommonCrawl-CreativeCommons},
  version = {1.1.0},
  year = {2025}
}
```


## Acknowledgments

- [TNO](https://www.tno.nl/nl/), who funded the work hours to accomplish this code. They intend to use (parts of) [the generated material](https://huggingface.co/datasets/BramVanroy/CommonCrawl-CreativeCommons) for the [GPT-NL project](https://gpt-nl.nl/).
- [Flemish Supercomputer Center](https://www.vscentrum.be/) for part of the compute under grant 2024-107
- Guilherme Penedo ([@guipenedo](https://huggingface.co/guipenedo)) and the rest of the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [datatrove](https://github.com/huggingface/datatrove) team for the help and insights
- ML6 and specifically Robin Van Craenenbroek for their [Fondant Creative Commons](https://github.com/ml6team/fondant-usecase-filter-creative-commons/tree/add-fondant-usecase-cc-image-extraction) filter for image datasets. While my approach is different, their code did serve as inspiration.

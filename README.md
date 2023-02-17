# Encouraging responsible reporting practices in the _Instructions to Authors_ of neuroscience and physiology journals: there is room to improve

Joanna Diong,<sup>1,2</sup> 
Elizabeth Bye,<sup>2,3</sup> 
Zoë Djajadikarta,<sup>2</sup> 
Annie A Butler,<sup>2,3</sup> 
Simon C Gandevia,<sup>2,4</sup> 
Martin E Héroux<sup>2,3</sup> 

1. School of Medical Sciences, Faculty of Medicine and Health, The University of Sydney
2. Neuroscience Research Australia (NeuRA)
3. School of Medical Sciences, University of New South Wales
4. Clinical School, University of New South Wales

## Suggested citation

Diong J, Bye E, Djajadikarta Z, Butler AA, Gandevia SC, Héroux ME (2023) 
Encouraging responsible reporting practices in the _Instructions to Authors_ of neuroscience and physiology journals:
there is room to improve.
PLOS One (in press).

## Protocol registration

The protocol for this study is stored in the Open Science Foundation (OSF) project folder [https://osf.io/wtkzy/][project_folder],
and registered on the OSF: [https://osf.io/w8zxj][rego]

## Data

Raw data are stored in **data/raw/**.

* JournalHomeGrid-final-2104221.csv
* crosscheck criteria - scores.csv
* JournalInstructionsT_DATA_LABELS_2022-02-23_1305.csv
* JournalInstructionsT_DATA_2022-04-04_1551.csv

## Code

Python code files (Python v3.9) were written by Joanna Diong.

### Python files

`script`: Main script to run analysis.

`proc`: Module containing functions used to clean data and plot figures.

### Running Python code

A reliable way to reproduce the analysis would be to run the code in an integrated development environment for Python (e.g. [PyCharm][pycharm]). 

Create a virtual environment and install dependencies. Using the Terminal (Mac or Linux, or PyCharm Terminal), 

```bash 
python -m venv env
```
Next, activate the virtual environment. 

For Mac or Linux, 

```bash
source env/bin/activate
```

For Windows, 

```bash
.\env\Scripts\activate
```

Then, install dependencies,

```bash
pip install -r requirements.txt
```

Download all files into a single folder and run `script.py`.

## Output

Output are generated and stored in **data/proc/**:

__CSV files__

* comments.csv
* data.csv
* data_journal_extguide.csv
* describe.csv
* varcodes_and_labels.csv

__Text file__

* results.txt

__Figure files__

* research_methods.png
* research_methods.svg
* results_figs.png
* results_figs.svg
* results_text.png
* results_text.svg
* statistical_methods.png
* statistical_methods.svg
* transparency.png
* transparency.svg


[project_folder]: https://osf.io/wtkzy/
[rego]: https://osf.io/w8zxj
[pycharm]: https://www.jetbrains.com/pycharm/promo/?gclid=Cj0KCQiAtqL-BRC0ARIsAF4K3WFahh-pzcvf6kmWnmuONEZxi544-Ty-UUqKa4EelnOxa5pAC9C4_d4aAisxEALw_wcB 

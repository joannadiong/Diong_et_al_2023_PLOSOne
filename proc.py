import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.rcParams.update({'font.size': 8.5})  # Fig 3
# matplotlib.rcParams.update({'font.size': 11})  # Fig 1, 2, 4, 5


def process_raw_data(path_data):
    # data files
    journal_citation_IF_data = 'JournalHomeGrid-final-2104221.csv'
    external_guidelines_data = 'crosscheck criteria - scores.csv'
    main_data_labs = 'JournalInstructionsT_DATA_LABELS_2022-02-23_1305.csv'
    main_data_vars = 'JournalInstructionsT_DATA_2022-04-04_1551.csv'

    # read data: citations, external guidelines, main dataset variable labels
    df_metrics = _read_citation_IF_data(path_data, journal_citation_IF_data)
    question_stems: list = _read_external_guidelines_data_get_question_stems(path_data, external_guidelines_data)
    df_extguide, questions = _read_external_guidelines_data(path_data, external_guidelines_data)
    df_labels, redcap_column_labels, journal_name_labels = _read_main_dataset_labels_get_labels(path_data, main_data_labs)

    # get journals and referenced external guidelines
    ext_guideline_name_labels_xlsx, ext_guideline_name_labels_redcap, ext_guideline_name_labels_posthoc = \
        _get_external_guidelines_and_exceptions(df_extguide, df_labels)
    print(f"External guidelines added posthoc: {ext_guideline_name_labels_posthoc}")
    _get_journals_and_referenced_external_guidelines(path_data, df_labels, journal_name_labels, ext_guideline_name_labels_redcap)

    # read data: main dataset variables; clean data
    df_main, comments_varnames = _read_main_dataset_varcodes(path_data, main_data_vars, redcap_column_labels, df_labels)
    df_main = _process_main_dataset(df_main, comments_varnames, journal_name_labels)
    df = _merge_metrics_and_main_datasets(path_data, df_metrics, df_main)

    return df, question_stems, questions


def process_cleaned_data(path_data, df, question_stems, questions):
    # df_qn = _get_min_and_max_criteria_scores(referenced_ext_guidelines, df_extguide, questions)

    # get processed data for full dataset
    criteria, dicts = _subset_processed_data(path_data, question_stems, questions, ref_doc=True)
    # get processed data for journals that do not reference any external document
    _, dicts_solo = _subset_processed_data(path_data, question_stems, questions, ref_doc=False)

    _write_variable_names_and_definitions_to_file(path_data, criteria, questions)
    _describe_data(path_data, df)

    categories = ['research_methods', 'statistical_methods', 'results_text', 'results_figs', 'transparency']
    for category in categories:
        _calc_journal_counts(path_data, category, dicts, ref_doc=True)
        _calc_journal_counts(path_data, category, dicts_solo, ref_doc=False)
        _plot_journal_counts(path_data, dicts[category]['df'], dicts[category]['xlabels'], category + '.png')
        _plot_journal_counts(path_data, dicts[category]['df'], dicts[category]['xlabels'], category + '.svg')


def _read_citation_IF_data(path_data: Path, journal_citation_IF_data: str):
    """Read journal list dataset for citation numbers, impact factor"""
    df_metrics = pd.read_csv(path_data / 'raw' / journal_citation_IF_data)
    df_metrics = df_metrics.rename(columns={'title': 'journal'})
    return df_metrics


def _read_external_guidelines_data_get_question_stems(path_data: Path, external_guidelines_data: str):
    """Read Crosscheck Criteria spreadsheet, only get question stems"""
    df = pd.read_csv(path_data / 'raw' / external_guidelines_data)[:0]
    superheaders = df.columns.values

    question_stems = []
    for i in superheaders:
        if i.split(':')[0] == 'Unnamed':
            pass
        else:
            question_stems.append(i)
    question_stems = question_stems[1:]

    return question_stems


def _read_external_guidelines_data(path_data: Path, external_guidelines_data: str):
    """Read Crosscheck Criteria spreadsheet,
    set indexes for columns (questions) and rows (guidelines)"""
    df = pd.read_csv(path_data / 'raw' / external_guidelines_data, header=1)
    external_guidelines = df['R: Required,   E: Encouraged,   N: No mention']
    questions: np.ndarray = df.columns.values[1:]

    df = df.set_index(external_guidelines).drop(columns=['R: Required,   E: Encouraged,   N: No mention'])
    df.index.name = 'index'

    return df, questions


def _read_main_dataset_labels_get_labels(path_data: Path, main_data_labs: str):
    """Read main dataset of variable label names,
    get dataframe, RedCap column header labels, journal name labels"""
    df_labels = pd.read_csv(path_data / 'raw' / main_data_labs)
    # drop duplicate: record 56 mislabeling has been fixed in RedCap
    df_labels = df_labels.drop(df_labels[df_labels['Record ID'] == 101].index)

    redcap_column_labels: np.ndarray = df_labels.columns.values
    journal_name_labels: np.ndarray = df_labels['Journal name'].values

    return df_labels, redcap_column_labels, journal_name_labels


def _get_external_guidelines_and_exceptions(df_extguide: pd.DataFrame, df_labels: pd.DataFrame):
    """Get names of external guidelines from Crosscheck Criteria, and those added posthoc"""
    # get names of external guidelines from Crosscheck Criteria spreadsheet
    guidelines_names_ccxlsx = df_extguide.index.values

    # get names of external guidelines from RedCap main dataset of variable label names:
    # get column values, then e.g. extract 'AGREE' from 'Name(s) of reference document(s) (choice=AGREE)'
    guidelines_names_labels_redcap = df_labels.loc[:, df_labels.columns.str.startswith('Name(s) of reference document(s)')].columns.values
    guidelines_names_redcap = []
    for i in guidelines_names_labels_redcap:
        name = i.split('=')[1].strip(')')
        guidelines_names_redcap.append(name)

    # get guidelines Crosscheck Criteria that are not listed in RedCap:
    # these were external guidelines added posthoc to RedCap because they were missed on first round of checking
    guidelines_in_ccxlsx_not_in_redcap = []
    [guidelines_in_ccxlsx_not_in_redcap.append(i) for i in guidelines_names_ccxlsx if i not in guidelines_names_redcap]

    return guidelines_names_ccxlsx, guidelines_names_labels_redcap, guidelines_in_ccxlsx_not_in_redcap


def _get_journals_and_referenced_external_guidelines(path_data: Path, df_labels: pd.DataFrame, journal_name_labels: np.ndarray, ext_guideline_name_labels_redcap: np.ndarray):
    """Get list of any external guidelines referenced by each journal;
    if a journal references at least 1 external guideline, it will have a min and max criteria score"""
    journals_referenced_guidelines = []
    for journal in journal_name_labels:
        journal_referenced_guidelines = []
        for guideline_name_label in ext_guideline_name_labels_redcap:
            # for each guideline, if that guideline is referenced for that journal, append to list
            if df_labels.loc[df_labels['Journal name'] == journal, [guideline_name_label]].values.flatten()[0] == 'Checked':
                guideline_name = guideline_name_label.split('=')[1].strip(')')
                journal_referenced_guidelines.append(guideline_name)
            # else just return the empty list
            else:
                pass
        # append all lists of journal referenced guidelines into a super-list
        journals_referenced_guidelines.append(journal_referenced_guidelines)

    df_journal_extguide = pd.DataFrame(zip(journal_name_labels, journals_referenced_guidelines), columns=['journal', 'guidelines'])
    df_journal_extguide = df_journal_extguide.sort_values(by=['journal'])

    # add external guidelines that were detected after the initial pass
    df_journal_extguide.loc[df_journal_extguide['journal'] == 'neurology-neuroimmunology & neuroinflammation', 'guidelines'].values[0].append('Cummings')
    sorted(df_journal_extguide.loc[df_journal_extguide['journal'] == 'neurology-neuroimmunology & neuroinflammation', 'guidelines'].values[0])

    df_journal_extguide.loc[df_journal_extguide['journal'] == 'neuron', 'guidelines'].values[0].append('STAR Methods')
    sorted(df_journal_extguide.loc[df_journal_extguide['journal'] == 'neuron', 'guidelines'].values[0])

    df_journal_extguide.loc[df_journal_extguide['journal'] == 'physiology', 'guidelines'].values[0].append('APS_Editorial')
    sorted(df_journal_extguide.loc[df_journal_extguide['journal'] == 'physiology', 'guidelines'].values[0])

    df_journal_extguide.loc[df_journal_extguide['journal'] == 'journal of physiology-london', 'guidelines'].values[0].append('Drummond')
    df_journal_extguide.loc[df_journal_extguide['journal'] == 'journal of physiology-london', 'guidelines'].values[0].append('Schultz')
    sorted(df_journal_extguide.loc[df_journal_extguide['journal'] == 'journal of physiology-london', 'guidelines'].values[0])

    df_journal_extguide.to_csv(path_data / 'proc' / 'data_journal_extguide.csv')

    # calling the referenced ext guidelines;
    # if no ext guideline is referenced, return empty list (considered False in Python)
    # df_extguidelines.loc[df_extguidelines['journal'] == 'brain research', 'guidelines'].values[0]

    # return df_journal_extguide


def _read_main_dataset_varcodes(path_data: Path, main_data_vars: str, redcap_column_labels: np.ndarray, df_labels: pd.DataFrame):
    """Read main dataset of variable codes,
    write variable codes and labels to file,
    write text of comments to file"""
    df_main = pd.read_csv(path_data / 'raw' / main_data_vars)
    df_main = df_main.drop(df_main[df_main['record_id'] == 101].index)  # drop duplicate: record 56 mislabeling has been fixed in RedCap

    # write variable codes and variable labels to file
    variables = df_main.columns.values
    df_varcodes_labels = pd.DataFrame({'names': variables, 'labels': redcap_column_labels})
    df_varcodes_labels.to_csv(path_data / 'proc' / 'varcodes_and_labels.csv')

    # write comments text to file
    criteria_varnames = ['prereg_trial', 'prereg_other', 'stat_tests', 'outcomes', 'alpha', 'exclude', 'missing', 'prior',
                         'var', 'mean', 'median', 'no_sem', 'pval', 'exact_p', 'precision', 'trend', 'posterior', 'fig_var',
                         'fig_raw_data', 'pub_data', 'pub_proc', 'pub_stat']
    comments_varnames = list(df_main.loc[:, df_main.columns.str.startswith('comment')].columns.values)
    dataframes_list = []
    for criterion, comment in zip(criteria_varnames, comments_varnames):
        d = df_main[[criterion, comment]]
        dataframes_list.append(d)
        # print(d)
    dataframes_list.append(df_main[['inconsistent', 'inconsistent_txt']])
    df_comments = pd.concat([df_labels[['Journal name', 'Auditor initials']]] + dataframes_list, axis=1)
    df_comments = df_comments.sort_values(by=['Journal name'])
    df_comments.to_csv(path_data / 'proc' / 'comments.csv')

    return df_main, comments_varnames


def _process_main_dataset(df_main: pd.DataFrame, comments_varnames: list, journal_name_labels: np.ndarray):
    # set columns to drop from main dataframe: external guideline indicators, comments
    external_guideline_varnames = []
    for i in list(range(1, 32, 1)):
        external_guideline_code = 'ref_doc_name___' + str(i)
        external_guideline_varnames.append(external_guideline_code)

    # 'journal' is journal codes: drop this column and replace with journal name labels
    df_main = df_main.drop(['auditor', 'journal'] + external_guideline_varnames + comments_varnames + ['audit_complete'], axis=1)
    df_main['journal'] = journal_name_labels

    return df_main


def _merge_metrics_and_main_datasets(path_data: Path, df_metrics: pd.DataFrame, df_main: pd.DataFrame):
    # merge metrics and full datasets
    df = pd.merge(left=df_metrics,
                  right=df_main,
                  how='outer',
                  on='journal')
    df = df.sort_values(by=['journal'])
    df.to_csv(path_data / 'proc' / 'data.csv')

    return df


def _get_min_and_max_criteria_scores(referenced_ext_guidelines: list, df_extguide: pd.DataFrame, questions: np.ndarray):
    """If a journal reference at least 1 external guideline,
    return the min and max values of the range of scores for that journal using
    e.g. referenced_ext_guidelines = ['AGREE', 'APS_Rigor', 'ARRIVE'];

    return as a dict() because there are 22 criteria, each having a min and max score;

    To call min value: df_qn[questions[0]].min"""
    df_ = df_extguide.loc[referenced_ext_guidelines]

    min_scores = []
    max_scores = []
    combined_scores = []
    for question in questions:
        # scores are R: Required, E: Encouraged, N: No mention
        # get minimum score: score is N if at least 1 N is present in any external guideline
        if 'N' in df_[question].values:
            min_score = 'N'
            min_scores.append(min_score)
        # get maximum score: score is R if at least 1 R is present in any external guideline
        if 'R' in df_[question].values:
            max_score = 'R'
            max_scores.append(max_score)
        # max score is E if at least 1 E is present in any external guideline
        elif 'E' in df_[question].values:
            max_score = 'E'
            max_scores.append(max_score)
        # max score is N if no R or E is present
        elif 'R' not in df_[question].values or 'E' not in df_[question].values:
            max_score = 'N'
            max_scores.append(max_score)

        # make dataclass of combined scores
        @dataclass
        class ExtGuidelineScores:
            """Hold min and max criteria scores across all external guidelines referenced in author instructions"""
            min: 'str'
            max: 'str'
        values = ExtGuidelineScores(min=min_score, max=max_score)
        combined_scores.append(values)

    # for a single journal, zip questions and combined scores into dict. To call min value: d[questions[0]].min
    df_qn = {}
    for i, j in zip(questions, list(range(len(questions)))):
        df_qn[i] = combined_scores[j]

    return df_qn


def _subset_processed_data(path_data: Path, question_stems: list, questions: np.array, ref_doc: bool=True):
    """Break up processed data into categories of different criteria, return"""
    df = pd.read_csv(path_data / 'proc' / 'data.csv')
    if not ref_doc:
        df = df[df['ref_doc'] == 0]

    # set variable names for each category of criteria
    criteria_research_methods = ['prereg_trial', 'prereg_other', 'stat_tests', 'outcomes']
    criteria_statistical_methods = ['alpha', 'exclude', 'missing', 'prior']
    criteria_results_text = ['var', 'mean', 'median', 'no_sem', 'pval', 'exact_p', 'precision', 'trend', 'posterior']
    criteria_results_figs = ['fig_var', 'fig_raw_data']
    criteria_transparency = ['pub_data', 'pub_proc', 'pub_stat']
    criteria = criteria_research_methods + criteria_statistical_methods + criteria_results_text + criteria_results_figs + criteria_transparency

    # set xlabels for plots
    xlabels_research_methods = ['prereg\ntrial', 'prereg\nother', 'stat\ntests', 'outcomes']
    xlabels_statistical_methods = ['alpha', 'exclude', 'missing', 'prior']
    xlabels_results_text = ['var', 'mean', 'median', 'no\nsem', 'pval', 'exact\np', 'precision', 'trend', 'posterior']
    xlabels_results_figs = ['var', 'raw data']
    xlabels_transparency = ['pub\ndata', 'pub\nproc', 'pub\nstat']

    # get questions within each category
    questions_research_methods = questions[:4]
    questions_statistical_methods = questions[4:8]
    questions_results_text = questions[8:17]
    questions_results_figs = questions[17:19]
    questions_transparency = questions[19:]

    # get question stems
    stem_research_methods = question_stems[0]
    stem_statistical_methods = question_stems[1]
    stem_results_text = question_stems[2]
    stem_results_figs = question_stems[3]
    stem_transparency = question_stems[4]

    # subset the main dataframe into dataframes for each category of criteria
    df_research_methods = df.iloc[:, np.r_[9, 10:14]]
    df_statistical_methods = df.iloc[:, np.r_[9, 14:18]]
    df_results_text = df.iloc[:, np.r_[9, 18:27]]
    df_results_figs = df.iloc[:, np.r_[9, 27:29]]
    df_transparency = df.iloc[:, np.r_[9, 29:32]]

    dict_research_methods = {'criteria': criteria_research_methods, 'xlabels': xlabels_research_methods, 'questions': questions_research_methods, 'stem': stem_research_methods, 'df': df_research_methods}
    dict_statistical_methods = {'criteria': criteria_statistical_methods, 'xlabels': xlabels_statistical_methods, 'questions': questions_statistical_methods, 'stem': stem_statistical_methods, 'df': df_statistical_methods}
    dict_results_text = {'criteria': criteria_results_text, 'xlabels': xlabels_results_text, 'questions': questions_results_text, 'stem': stem_results_text, 'df': df_results_text}
    dict_results_figs = {'criteria': criteria_results_figs, 'xlabels': xlabels_results_figs, 'questions': questions_results_figs, 'stem': stem_results_figs, 'df': df_results_figs}
    dict_transparency = {'criteria': criteria_transparency, 'xlabels': xlabels_transparency, 'questions': questions_transparency, 'stem': stem_transparency, 'df': df_transparency}
    dicts = {'research_methods': dict_research_methods, 'statistical_methods': dict_statistical_methods, 'results_text': dict_results_text, 'results_figs': dict_results_figs, 'transparency': dict_transparency}

    return criteria, dicts


def _write_variable_names_and_definitions_to_file(path_data: Path, criteria: list, questions: np.array):
    """Write header"""
    file = path_data / 'proc' / 'results.txt'
    open(file, 'w').close()
    with open(file, 'a') as file:
        for criterion, question in zip(criteria, questions):
            file.write('\n[{}]: {}'.format(criterion, question))


def _describe_data(path_data: Path, df: pd.DataFrame):
    df[['impact_factor', 'total_cites']].describe().to_csv(path_data / 'proc' / 'describe.csv')


def _calc_journal_counts(path_data: Path, category: str, dicts: dict, ref_doc: bool=True):
    """Get text file of numeric counts of journals that satisfy the criteria"""
    file = path_data / 'proc' / 'results.txt'
    labels = {1: 'a) not mentioned', 2: 'b) encouraged', 3: 'c) required'}

    # write count data to file
    counter = 0
    if category == 'research_methods':
        qstart = 1
    elif category == 'statistical_methods':
        qstart = 5
    elif category == 'results_text':
        qstart = 9
    elif category == 'results_figs':
        qstart = 18
    elif category == 'transparency':
        qstart = 20
    with open(file, 'a') as file:
        if ref_doc:
            file.write('\n\n\nSection {}'.format(dicts[category]['stem']))
        for criterion, question in zip(dicts[category]['criteria'], enumerate(dicts[category]['questions'])):
            if ref_doc:
                file.write('\n\nQ{}. {}\n'.format(question[0] + qstart, question[1]))
                file.write(dicts[category]['df'][criterion].apply(lambda label: labels[label]).value_counts().sort_index().to_string())
            if not ref_doc:
                if counter == 0:
                    file.write('\n\n> SUBSET: journals that did not reference external guidelines (n, %):')
                file.write('\n\nQ{}. {}\n'.format(question[0] + qstart, question[1]))
                file.write(dicts[category]['df'][criterion].apply(lambda label: labels[label]).value_counts().sort_index().to_string())
                # file.write('\n\n')
                # file.write(dicts[category]['df'][criterion].apply(lambda label: labels[label]).value_counts(normalize=True).sort_index().to_string())
                counter += 1


def _plot_journal_counts(path_data: Path, df: pd.DataFrame, xlabels: list, figname: str):
    """Plot journals that satisfy the criteria: df = dicts['research_methods']['df']"""
    colors = {1: 'red', 2: 'orange', 3: 'green'}

    patch_r = mpatches.Patch(color='red', label='Not mentioned')
    patch_o = mpatches.Patch(color='orange', label='Encouraged')
    patch_g = mpatches.Patch(color='green', label='Required')
    legend_handles = [patch_r, patch_o, patch_g]

    # Plot Pandas stacked bar plots: full sample and subset of ref_doc==0, in %
    # vars = df.columns.values[1:]
    # ax = df[vars].transform(np.sort).apply(lambda count: count.value_counts()).transpose(). \
    #     plot(kind='bar', position=1.1, width=0.2, color=colors, alpha=1, stacked=True, rot=0)  # colormap='RdYlGn'
    # df[vars][df['ref_doc'] == 0].transform(np.sort).apply(lambda count: count.value_counts(normalize=True) * 100).transpose(). \
    #     plot(ax=ax, kind='bar', position=-0.1, width=0.2, color=colors, alpha=0.5, stacked=True, rot=0)
    # plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1))

    # Plot Pandas stacked bar plots: full sample and subset of ref_doc==0, in counts
    vars = df.columns.values[1:]
    ax = df[vars].transform(np.sort).apply(lambda count: count.value_counts()).transpose(). \
        plot(kind='bar', position=1.1, width=0.2, color=colors, alpha=1, stacked=True, rot=0)
    df[vars][df['ref_doc'] == 0].transform(np.sort).apply(lambda count: count.value_counts()).transpose(). \
        plot(ax=ax, kind='bar', position=-0.1, width=0.2, color=colors, alpha=0.5, stacked=True, rot=0)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1))

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
    plt.ylabel('No. of journals')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    file = path_data / 'proc' / figname
    plt.savefig(file, dpi=300)
    plt.close()

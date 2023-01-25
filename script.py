from pathlib import Path
import proc

LOCAL = Path('/home/joanna/Dropbox/Projects/auGuidelines/data/')
REPO = Path('./data/')
path_data = REPO

# PROCESS RAW DATA
df, question_stems, questions = proc.process_raw_data(path_data)

# PROCESS CLEANED DATA
proc.process_cleaned_data(path_data, df, question_stems, questions)


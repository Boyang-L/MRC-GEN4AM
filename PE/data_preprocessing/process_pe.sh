echo 'Load data'
cat data/conll/Essay_Level/train.dat \
    data/conll/Essay_Level/test.dat \
    data/conll/Essay_Level/dev.dat > \
    data/conll/Essay_Level/all.dat
cat data/conll/Paragraph_Level/train.dat \
    data/conll/Paragraph_Level/test.dat \
    data/conll/Paragraph_Level/dev.dat > \
    data/conll/Paragraph_Level/all.dat
python load_text_essays.py \
    --tag \
    --data-path data/ \
    > data/PE_token_level_data.tsv
python pe_token_to_pe_df.py

source ~/environments/pytorch1/bin/activate

python experiment.py data/ontonotes_dev_info_NN-PRP.csv --cuda --method gradient --no_overwrite --combine no --out output/overnight3 --n_items 500 --factors coref --track noun,pronoun --prefix COREF
python experiment.py data/ontonotes_dev_info_NN-PRP.csv --cuda --method attention --no_overwrite --combine no --out output/overnight3 --n_items 500 --factors coref --track noun,pronoun --prefix COREF
python experiment.py data/ontonotes_dev_info_NN-PRP.csv --cuda --method gradient --no_overwrite --combine chain --out output/overnight3 --n_items 500 --factors coref --track noun,pronoun --prefix COREF

python experiment.py data/en_gum-ud-dev_open-closed.csv --method gradient --no_overwrite --combine no --out output/overnight3 --balance --cuda --balance --n_items 500 --track ... --prefix OPEN-CLOSED
python experiment.py data/en_gum-ud-dev_open-closed.csv  --method attention --no_overwrite --combine no --out output/overnight3 --balance --cuda --balance --n_items 500 --track ... --prefix OPEN-CLOSED
python experiment.py data/en_gum-ud-dev_open-closed.csv  --method gradient --no_overwrite --combine chain --out output/overnight3 --balance --cuda --balance --n_items 500 --track ... --prefix OPEN-CLOSED

python experiment.py data/en_gum-ud-dev_POS.csv --method gradient --no_overwrite --combine no --balance --out output/overnight3 --cuda --track ... --n_items 500 --prefix POS
python experiment.py data/en_gum-ud-dev_POS.csv --method gradient --no_overwrite --combine chain --balance --out output/overnight3 --cuda --track ... --n_items 500 --prefix POS
python experiment.py data/en_gum-ud-dev_POS.csv --method attention --no_overwrite --combine no --balance --out output/overnight3 --cuda --track ... --n_items 500 --prefix POS



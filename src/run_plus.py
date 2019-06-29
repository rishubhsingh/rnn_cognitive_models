from agreement_acceptor import GramHalfPlusSentence
import filenames

pvn = GramHalfPlusSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10)
pvn.pipeline(train=True, load=False, epochs=10, model_prefix='lstm_plus10', data_name='plus10')
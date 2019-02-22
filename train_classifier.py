from fastai.text import *

path = Path("data/")

batch_size=16

data = (
    TextList.from_folder(path)
    .filter_by_folder(include=['train', 'test', 'unsup'])
    .random_split_by_pct(0.1)
    .label_for_lm()
    .databunch(bs=batch_size)
)

classifier_data = (
    TextList.from_folder(path, vocab=data.vocab)
    .split_by_folder(valid='test')
    .label_from_folder(classes=['neg', 'pos'])
    .databunch(bs=batch_size)
)

learn = text_classifier_learner(classifier_data, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('encoder-5-epochs')

print("Stage 2:")
print("-" * 20)
# resuming from stage-2
learn.load('classifier-stage-1')
learn.freeze_to(-2)
learn.fit_one_cycle(5, slice(2e-4, 1e-2))
learn.save('classifier-stage-2-v2')

print("Stage 3:")
print("-" * 20)
# training the last 3 layers
learn.load('classifier-stage-2-v2')
learn.freeze_to(-3)
learn.fit_one_cycle(3, slice(1e-4, 5e-3))
learn.save('classifier-stage-3-v2')

print("Stage 4:")
print("-" * 20)
learn.load('classifier-stage-3-v2')
learn.unfreeze()
learn.fit_one_cycle(5, slice(2e-5, 1e-3))
learn.save('classifier-stage-4-v2')

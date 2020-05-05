import ctgan
import time

data, discrete_columns = ctgan.utils.load_demo()
start = time.time()
model = ctgan.synthesizer.CTGANSynthesizer()
model.train(data, discrete_columns, epochs=200)
samples = model.sample(data.shape[0])
samples.to_csv('tests/tensorflow.csv')
end = time.time() - start
print(end)

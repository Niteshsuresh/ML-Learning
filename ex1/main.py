from brain import Brain
b = Brain();
b.classify('data.csv');
b.test([6,148,72,35,0,33.6,0.627,50]);
b.test([1,85,66,29,0,26.6,0.351,31]);
b.test([8,183,64,0,0,23.3,0.672,32]);
b.test([1,89,66,23,94,28.1,0.167,21]);
b.accuracy();

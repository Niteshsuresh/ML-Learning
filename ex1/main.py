from brain import Brain
b = Brain();
b.classify('data.csv');
b.test([6,148,72,35,0,33.6,0.627,50]);
b.test([1,85,66,29,0,26.6,0.351,31]);
b.accuracy();

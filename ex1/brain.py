from file_helper import load_csv
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain import TanhLayer
from pybrain.structure.modules import SoftmaxLayer
import pickle


def get_path(file_name):
    return file_name
class Brain():
    def add_patients_data_to_train(self,file_name):
        patient_ls = load_csv(file_name);
        max_len = len(patient_ls[0]);
        train_data_size = (len(patient_ls))/2;
        self.train_data = patient_ls[0:train_data_size];
        self.test_data = patient_ls[train_data_size: len(patient_ls)];
        for test_data in self.test_data:
            self.t_ds.addSample(test_data[0: max_len - 1], test_data[-1]);
        for patient in self.train_data:
            self.ds.addSample(patient[0: max_len -1], patient[-1]);

    def save(self, file_name="classifier.brain"):
        with open(get_path(file_name), 'wb') as file_pointer:
            pickle.dump(self.classifier_neural_net, file_pointer)
        
    def load(self, file_name="classifier.brain"):
        with open(get_path(file_name), 'rb') as file_pointer:
            self.classifier_neural_net = pickle.load(file_pointer)
    
    def train(self):
        self.ds._convertToOneOfMany();
        self.trainer.trainEpochs(10);

    def accuracy(self):
        if len(self.test_data) == 0:
            print "No data_sets found. Maybe you loaded the classifier from a file?"
            return
        tstresult = percentError(
                    self.trainer.testOnClassData(dataset=self.t_ds),self.t_ds['class'])
        print "epoch: %4d" % self.trainer.totalepochs, \
                              "trainer error: %5.2f%%" % tstresult, \
                                            "trainer accuracy: %5.2f%%" % (100-tstresult)

    def classify(self,file_name):
        #self.load();
        self.t_ds = ClassificationDataSet(8,1,nb_classes=2);
        self.ds = ClassificationDataSet(8, 1, nb_classes=2);
        self.classifier_neural_net = buildNetwork(8, 30, 2, outclass=SoftmaxLayer, hiddenclass=TanhLayer)
        self.trainer = BackpropTrainer(self.classifier_neural_net,self.ds)
        self.add_patients_data_to_train(file_name);  
        self.train();
        self.save();
    
    def test(self, array):
        score = self.classifier_neural_net.activate(array);
        score = max(xrange(len(score)), key=score.__getitem__)
        print(score);


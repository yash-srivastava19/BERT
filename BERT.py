"""Resources are Exhausted !  """
import tensorflow.keras as keras
import tensorflow as tf
import os,math,numpy,pandas,glob,re
from dataclasses import dataclass

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# This method is much better than what we do now.
@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8
    FF_DIM = 128
    NUM_LAYERS = 1

config = Config()

def GetTextListFromFiles(files):
    textList = []
    for name in files:
        with open(name) as f:
            for line in f:
                textList.append(line)

    return textList

def GetDataFromTextFiles(folder_name):
    pos_files = glob.glob("aclImdb/{}/pos/*.txt".format(folder_name))
    pos_texts = GetTextListFromFiles(pos_files)

    neg_files = glob.glob("aclImdb/{}/neg/*.txt".format(folder_name))
    neg_texts = GetTextListFromFiles(neg_files)

    df = pandas.DataFrame(
        {
            "review"    :pos_texts + neg_texts,
            "sentiment" :[0]*len(pos_texts) + [1]*len(neg_texts),
        }
    )

    df = df.sample(len(df)).reset_index(drop = True)
    return df

TrainDF = GetDataFromTextFiles("train")
TestDF  = GetDataFromTextFiles("test")
AllData = TrainDF.append(TestDF)

# TextVectorization is used to vectorize text into token ids.

def CustomStandardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase,"<br />"," ")
    return tf.strings.regex_replace(stripped_html,"[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"),"")


def GetVectorizeLayer(Texts,VocabSize,MaxSeq,SpecialTokens = ["[MASK]"]):
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens = VocabSize,
        output_mode = "int",
        standardize = CustomStandardization,
        output_sequence_length = MaxSeq,
    )

    vectorize_layer.adapt(Texts)

    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2:VocabSize - len(SpecialTokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)

    return vectorize_layer


vectorize_layer = GetVectorizeLayer(
    AllData.review.values.tolist(),
    config.VOCAB_SIZE,
    config.MAX_LEN,
    SpecialTokens = ["[mask]"])

mask_token_id = vectorize_layer(["mask"]).numpy()[0][0]

def Encode(texts):
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()


def GetMaskedInputAndLabels(Encoded_Texts):
    # 15% BERT masking
    inp_mask = numpy.random.rand(*Encoded_Texts.shape) < 0.15

    # Do not mask special tokens
    inp_mask[Encoded_Texts <= 2] = False

    labels = -1 * numpy.ones(Encoded_Texts.shape,dtype = int)

    labels[inp_mask] = Encoded_Texts[inp_mask]

    encoded_texts_masked = numpy.copy(Encoded_Texts)

    inp_mask_2_mask = inp_mask & (numpy.random.rand(*Encoded_Texts.shape) < 0.90)

    encoded_texts_masked[inp_mask_2_mask] = mask_token_id
    
    inp_mask_2_random = inp_mask_2_mask & (numpy.random.rand(*Encoded_Texts.shape) < 1/9)
    encoded_texts_masked[inp_mask_2_random] = numpy.random.randint(3,mask_token_id,inp_mask_2_random.sum())

    sample_weights = numpy.ones(labels.shape)
    sample_weights[labels == -1] = 0

    yLabels = numpy.copy(Encoded_Texts)

    return (encoded_texts_masked,yLabels,sample_weights)


xTrain = Encode(TrainDF.review.values)
yTrain = TrainDF.sentiment.values
TrainClassifierDS = (tf.data.Dataset.from_tensor_slices((xTrain,yTrain)).shuffle(1000).batch(config.BATCH_SIZE))

xTest = Encode(TestDF.review.values)
yTest = TestDF.sentiment.values
TestClassifierDS = (tf.data.Dataset.from_tensor_slices((xTest,yTest)).batch(config.BATCH_SIZE))

TestRawClassifierDS = tf.data.Dataset.from_tensor_slices((TestDF.review.values,yTest)).batch(config.BATCH_SIZE)

xAllReview = Encode(AllData.review.values)

xMaskedTrain,yMaskedLabels,SampleWeights = GetMaskedInputAndLabels(xAllReview)

mlmDS = tf.data.Dataset.from_tensor_slices((xMaskedTrain,yMaskedLabels,SampleWeights)).shuffle(1000).batch(config.BATCH_SIZE)


def BERT_Module(query,key,value,i):
    attention_output = keras.layers.MultiHeadAttention(
        num_heads = config.NUM_HEAD,
        key_dim = config.EMBED_DIM // config.NUM_HEAD,)(query,key,value)

    attention_output = keras.layers.Dropout(0.1)(attention_output)

    attention_output = keras.layers.LayerNormalization(
        epsilon = 1e-6)(query + attention_output)

    ffn = keras.Sequential([
        keras.layers.Dense(config.FF_DIM, activation = "relu"),
        keras.layers.Dense(config.EMBED_DIM),
    ])

    ffn_output = ffn(attention_output)
    ffn_output = keras.layers.Dropout(0.1)(ffn_output)

    seq_output = keras.layers.LayerNormalization(epsilon = 1e-6)(attention_output + ffn_output)

    return seq_output


def GetPosEncodingMatrix(MaxLen,D_Emb):
    pos_enc = numpy.array([
        [pos / numpy.power(10000,2*(j//2)/D_Emb) for j in range(D_Emb)] if pos else numpy.zeros(D_Emb)
        for pos in range(MaxLen)])
    
    pos_enc[1:,0::2] = numpy.sin(pos_enc[1:,0::2])
    pos_enc[1:,1::2] = numpy.cos(pos_enc[1:,1::2])

    return pos_enc

loss_function = keras.losses.SparseCategoricalCrossentropy(reduction = keras.losses.Reduction.NONE)
loss_tracker = keras.metrics.Mean(name = "loss")

class MaskedLanguageModel(keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features,labels,sample_weight = inputs 
        else:
            features,labels = inputs
            sample_weight = None 

        with tf.GradientTape() as tape:
            predictions = self(features,training = True)
            loss = loss_function(labels,predictions,sample_weight = sample_weight)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss,trainable_vars)

        self.optimizer.apply_gradients(zip(gradients,trainable_vars)) 

        loss_tracker.update_state(loss,sample_weight = sample_weight)

        return {"Loss": loss_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker]


def CreateMaskedLanguageBERTModel():
    inputs = keras.layers.Input((config.MAX_LEN,),dtype = tf.int64)
    
    word_embeds = keras.layers.Embedding(config.VOCAB_SIZE,config.EMBED_DIM)(inputs)
    
    pos_embeds = keras.layers.Embedding(
        input_dim = config.MAX_LEN,
        output_dim = config.EMBED_DIM, 
        weights = [GetPosEncodingMatrix(config.MAX_LEN,config.EMBED_DIM)])(tf.range(0,config.MAX_LEN))
    
    embeds = word_embeds + pos_embeds

    encoder_output = embeds

    for i in range(config.NUM_LAYERS):
        encoder_output = BERT_Module(encoder_output,encoder_output,encoder_output,i)
    
    mlm_output = keras.layers.Dense(config.VOCAB_SIZE,activation = "softmax")(encoder_output)

    mlm_model = MaskedLanguageModel(inputs,mlm_output)
    mlm_model.compile(optimizer = keras.optimizers.Adam(config.LR))

    return mlm_model

ID2Token = dict(enumerate(vectorize_layer.get_vocabulary()))
Token2ID = {v:k for k,v in ID2Token.items()}

sample_tokens = vectorize_layer(["I have watched this [mask] and it was awesome"])

class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self,sample_tokens,top_k = 5):
        self._sample_tokens = sample_tokens
        self._k = top_k

    def decode(self,tokens):
        return " ".join(ID2Token[t] for t in tokens if t)

    def ConvertIDs2Tokens(self,_id):
        return ID2Token[_id]

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self._sample_tokens)

        maskedIndex = numpy.where(self._sample_tokens == mask_token_id)
        maskedIndex = maskedIndex[1]
        maskedPreds = preds[0][maskedIndex]

        topIndices = maskedPreds[0].argsort()[self._k:][::-1]
        values = maskedPreds[0][topIndices]


        for i in range(len(topIndices)):
            p,v = topIndices[i],values[i]
            tokens = numpy.copy(sample_tokens[0])
            tokens[maskedIndex[0]] = p

            result = {
                "input_text":self.decode(sample_tokens[0].numpy()),
                "prediction":self.decode(tokens),
                "probability":v,
                "predicted mask token":self.ConvertIDs2Tokens(p),}
            
            print(result)
        
gen_callback = MaskedTextGenerator(sample_tokens.numpy())

bert_masked_model = CreateMaskedLanguageBERTModel()

# Debug:bert_masked_model.summary()

bert_masked_model.fit(mlmDS,epochs = 5,callbacks = [gen_callback])

bert_masked_model.save("SavedModels/BertMLMIMDB.h5")
    

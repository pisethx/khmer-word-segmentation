import os
import psutil
import sklearn_crfsuite
import urllib.request


import glob
import re

import pickle
from sklearn.model_selection import train_test_split
import pycrfsuite


from util.helper import (
    get_correction_list,
    correct_str,
    cleanup_str,
    is_khmer_char,
    is_start_of_kcc,
    seg_kcc,
    gen_kcc_with_label,
    get_type,
    is_no_space,
    kcc_type,
    kcc_to_features,
    generate_kccs_label_per_phrase,
    create_kcc_features,
    create_labels_from_kccs,
    seg_char,
    gen_char_with_label,
    extract_features,
)

MODEL_PICKLE = "util/crf.model.pickle"


def train_model():
    sizes = ["100", "500", "1000", "5000", "10000"]
    docsize = sizes[0]
    data_dir = "kh_data_" + docsize
    file_name = data_dir + "_200b.zip"

    system_file = os.path.join("util", file_name)

    base_url = "https://github.com/phylypo/segmentation-crf-khmer/raw/master/data/"
    url = base_url + file_name

    urllib.request.urlretrieve(url, system_file)

    system_dir = os.path.join("util", data_dir)
    # remove previous existing directory for rerun
    os.system("rm -r {}".format(system_dir))
    os.system("unzip {} -d util/ | tail -10".format(system_file))

    path = system_dir + "/*_seg_200b.txt"  #  earlier format: *_seg.txt
    files = glob.glob(path)

    # global variables that use through out
    seg_text = []
    orig_text = []
    doc_ids = []

    for file in files:
        filenum = re.search(r"\d+_", file).group(0)
        doc_ids.append(filenum.replace("_", ""))
        f = open(file, "r")
        lines = f.readlines()
        f.close()
        seg_text.append(lines)

        # limit to 9K to avoid memory issue
        if len(seg_text) >= 9000:
            break  # kcc=9000, char:5000, crash char on 7000

        # read orig text -- comment out (10K docs which do not have orig text)
        f = open(file.replace("_seg_200b.txt", "_orig.txt"), "r")
        lines = f.readlines()
        f.close()
        orig_text.append(lines)

    idx = 0
    spacer = "\u2022"

    # setup training data using seg
    sentences = list()
    for i, text in enumerate(seg_text):
        for sentence in text:
            sentences.append(cleanup_str(sentence))

    # testing some text
    t1 = "យោងតាមប្រភពព័ត៌មានបានឱ្យដឹងថា កាលពីពេលថ្មីៗនេះក្រុមចក្រភពអង់គ្លេស Royal Marines ដែលមានមូលដ្ឋាននៅ Gibraltar បានរឹបអូសយកនាវាដឹកប្រេងឆៅរបស់អ៊ីរ៉ង់ដែលធ្វើដំណើរទៅកាន់រោងចក្រចម្រាញ់ប្រេងនៅក្នុងប្រទេសស៊ីរី ដោយក្រុងឡុងដ៍អះអាងថា ការរឹបអូសត្រូវបានគេសំដៅអនុវត្ត៕"
    t2 = "ខែThis is a test. N.B. ខែ? Test?"
    t3 = "នៅរសៀលថ្ងៃទី២២ ខែ កក្កដា ឆ្នាំ២០១៩ ឯកឧត្តម គួច ចំរើន អភិបាលខេត្តព្រះសីហនុ"
    t4 = "This. 11,12 ២២.២២២.២២២,២២"
    t5 = " ក "

    # test label
    ts = "This is a test"
    ts = "នៅ រសៀល ថ្ងៃ ទី ២២ ខែ កក្កដា ឆ្នាំ ២០១៩ ។"
    # ts = cleanup_str(ts)
    kccs = seg_kcc(ts)
    kl = gen_kcc_with_label(ts)

    # test
    ts = "នៅ រសៀល ថ្ងៃ ទី ២២ ខែ កក្កដា ឆ្នាំ ២០១៩ ។"
    # ts = '\u200bប្រភព\u200b៖ \u200bKenh\u200b \u200b14\u200b \u200bអត្ថបទ\u200bដោយ\u200b៖ \u200bTrassi\u200b \u200b'
    ts = cleanup_str(ts)
    kccs = seg_kcc(ts)
    kccs_label = gen_kcc_with_label(ts)  # only need for training

    fs = create_kcc_features(kccs)
    labels = create_labels_from_kccs(kccs_label)
    # create kccs, feature and labels for training for KCC based
    kccs_label = []
    kccs_only = []
    labels = []
    i = 0
    for sen in sentences:
        kcc_with_label = gen_kcc_with_label(sen)
        kccs_label.append(kcc_with_label)
        kccs_only.append(seg_kcc(sen))
        labels.append(create_labels_from_kccs(kcc_with_label))
        i = i + 1

    # test label
    ts = "This is a test"
    ts = "នៅ រសៀល ថ្ងៃ ទី ២២ ខែ កក្កដា ឆ្នាំ ២០១៩ ។"
    kccs = seg_char(ts)
    kl = gen_char_with_label(ts)

    # create kccs and its labels for training for char based
    chars_label = []
    chars_only = []
    labels_char = []

    for sen in sentences:
        chars_with_label = gen_char_with_label(sen)
        chars_label.append(chars_with_label)
        chars_only.append(seg_char(sen))
        labels_char.append(create_labels_from_kccs(chars_with_label))

    ts = "នៅ រសៀល ថ្ងៃ ទី ២២ ខែ កក្កដា ឆ្នាំ ២០១៩ ។"
    char = seg_char(ts)
    kl = gen_char_with_label(ts)

    # use list of character based chars_only to create feature
    X_char = [create_kcc_features(kcc_line) for kcc_line in chars_only]
    y_char = labels_char

    X_train_char, X_test_char, y_train_char, y_test_char = train_test_split(
        X_char, y_char, test_size=0.20, random_state=1
    )

    # used 19GB/25 for 5K docs

    crf_char = sklearn_crfsuite.CRF(
        algorithm="lbfgs",  #'l2sgd', #'lbfgs',
        c1=0.015,  # 0.1 not need for 'l2sgd'
        c2=0.0037,
        max_iterations=100,  # 100
        all_possible_transitions=True,
        verbose=False,
    )
    crf_char.fit(X_train_char, y_train_char)

    result_char = crf_char.score(X_train_char, y_train_char)

    result_char = crf_char.score(X_test_char, y_test_char)

    # kcc_data is list of kcc
    X = [extract_features(kcc_line) for kcc_line in kccs_only]
    y = labels

    indices = range(len(X))
    X_train, X_test, y_train, y_test, X_train_idx, X_test_idx = train_test_split(
        X, y, indices, test_size=0.2, random_state=1
    )  # 0.20
    kcc_list = [item for s in kccs_only for item in s]
    kcc_set = set(kcc_list)

    # clear memory for heavy run on unneeded X,y --19.47GB

    # process = psutil.Process(os.getpid())
    # print("Memory used before:", process.memory_info().rss)  # in bytes

    # process = psutil.Process(os.getpid())
    # print("Memory used after:", process.memory_info().rss)  # in bytes

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",  # options: 'l2sgd', 'lbfgs',
        c1=0.0418,  # 0.015, # not applicable for 'l2sgd'
        c2=0.00056,  # 0.0037,
        max_iterations=100,  # 100,
        all_possible_transitions=True,
        verbose=True,
    )
    crf.fit(X_train, y_train)

    trainer = pycrfsuite.Trainer(verbose=False)

    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # Set the parameters of the model
    trainer.set_params(
        {
            # coefficient for L1 penalty
            "c1": 0.1,  # 0.1
            # coefficient for L2 penalty
            "c2": 0.1,  # 0.01
            # maximum number of iterations
            "max_iterations": 200,  # 200,
            # whether to include transitions that
            # are possible, but not observed
            "feature.possible_transitions": True,
        }
    )

    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    trainer.train("util/crf.model")

    # save model to disk for further run
    pickle.dump(crf, open(MODEL_PICKLE, "wb"))

    # load the model from disk
    loaded_model = pickle.load(open(MODEL_PICKLE, "rb"))
    result = loaded_model.score(X_train, y_train)

    t = "ចំណែកជើងទី២ នឹងត្រូវធ្វើឡើងឯប្រទេសកាតា៕"
    t_correct = "ចំណែក ជើង ទី ២ នឹង ត្រូវ ធ្វើឡើង ឯ ប្រទេស កាតា ៕ "
    skcc = seg_kcc(t)

    features = create_kcc_features(skcc)
    pred = loaded_model.predict([features])

    separator = "-"
    tkcc = []
    for k in features:
        tkcc.append(k["kcc"])

    complete = ""
    for i, p in enumerate(pred[0]):
        if p == "1":
            complete += separator + tkcc[i]
        else:
            complete += tkcc[i]
    complete = complete.strip(separator)
    complete = complete.replace(separator + " " + separator, " ")


# output predicted -- give a string to be split by newline and sentences
def segment(str, spacer=" "):

    if not os.path.exists(MODEL_PICKLE):
        train_model()

    loaded_model = pickle.load(open(MODEL_PICKLE, "rb"))
    crf = loaded_model

    complete = ""
    for sen in str.split("\n"):
        if sen.strip() == "":
            continue
        sen = sen.replace(u"\u200b", "")
        kccs = seg_kcc(sen)
        features = create_kcc_features(kccs)
        prediction = crf.predict([features])

        for i, p in enumerate(prediction[0]):
            if p == "1":
                complete += spacer + kccs[i]
            else:
                complete += kccs[i]
        complete += "\n"

    complete = complete.replace(spacer + " ", " ").replace(
        " " + spacer, " "
    )  # no 200b before or after space

    return complete[:-1]

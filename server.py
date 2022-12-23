import os
import pickle
import pandas as pd
from flask import Flask, request, render_template,flash,redirect,session,abort,jsonify
from datetime import datetime
from analytics import write_to_csv_departments
from analytics import get_counts,get_tables,get_titles
from vncorenlp import VnCoreNLP
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer
import threading
import re


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
def model_fn(model_dir):
    print("model_fn_sentiment")
    return torch.jit.load(os.path.join(model_dir, 'model_sentiment.pt'), map_location=device)

def model_fn_topic(model_dir):
    print("model_fn_topic")
    return torch.jit.load(os.path.join(model_dir, 'model_topic.pt'), map_location=device)

def infer(model, tokenizer,  input_text):
  with VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') as rdrsegmenter:
    input_text = ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(input_text)])

  max_sequence_length = 32
  input_dict = tokenizer.encode_plus(
            input_text,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
  inputs = {
      "input_ids": torch.tensor(input_dict["input_ids"]).to(device),
      "attention_mask": torch.tensor(input_dict["attention_mask"]).to(device),
  }
  model.to(device)
  model.eval()
  with torch.no_grad():
    predictions, *_ = model(**inputs)
    predictions = torch.argmax(predictions, dim=1).flatten()
    predictions = predictions.detach().cpu().numpy()
  # trainer = Trainer(model=model)
  # predictions, *_ = trainer.predict(inputs)
  # print(predictions)

  return predictions

model_dir = "model"
modelA = model_fn(model_dir)
modelB = model_fn_topic(model_dir)
def normalize_text(text):
    # Remove các ký tự kéo dài: vd: đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
    # Chuyển thành chữ thường
    text = text.lower()
    # chuẩn hóa tiếng việt
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố', 'ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề', 'ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ', 'aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ', 'ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        'ô kêi': u'được', 'okie': u'được', 'o kê': u'được',
        'okey': u'được', 'ôkê': u'được', 'oki': u'được', 'oke': u'được', 'okay': u'được', 'okê': u'được',
        'tks': u'cám ơn', 'thks': u'cám ơn', 'thanks': u'cám ơn', 'ths': u'cám ơn', 'thank': u'cám ơn',
        'kg': u'không', 'not': u'không', u'kg': u'không', ' k ': u' không ', ' kh ': u' không ',
        'kô': u'không', 'hok': u'không', 'kp': u'không phải', u'kô': u'không', 'ko': u'không',
        u'ko': u'không', u' k ': u' không ', 'khong': u'không', u'hok': u'không',
        'sấu': u'xấu', 'gut': u'tốt', u'tot': u'tốt', u'nice': u'tốt', 'perfect': 'rất tốt',
        'bt': u'bình thường','time': u'thời gian', 'qá': u'quá', u'ship': u'giao hàng','max' : u'rất',
        u'm': u'mình', u'mik': u'mình', 
        	':)':'colonsmile',
            ':(' : 'colonsad',
            '@@' : 'colonsurprise',
            '<3' : 'colonlove',
            ':d' : 'colonsmilesmile',
            ':3' : 'coloncontemn',
            ':v' : 'colonbigsmile',
            ':_' : 'coloncc',
            ':p' : 'colonsmallsmile',
            '>>' : 'coloncolon',
            ':">': 'colonlovelove',
            '^^' :'colonhihi',
            ':' : 'doubledot',
            ":'(" : 'colonsadcolon',
            ':’(' : 'colonsadcolon',
            ':@' : 'colondoublesurprise',
            'v.v' : 'vdotv',
            '...' : 'dotdotdot',
            '.': ' .',
            ',':' ,',
         }
    for k, v in replace_list.items():
        text = text.replace(k, v)
        # remove nốt những ký tự thừa thãi
    text = text.replace(u'"', u' ')
    text = text.replace(u'️', u'')
    text = text.replace('🏻', '')
    return text

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'admin' and request.form['username'] == 'admin':
        session['logged_in'] = True
        return root()
    elif (request.form['password'] == 'student1' and request.form['username'] == 'student1') or (request.form['password'] == 'student2' and request.form['username'] == 'student2') or (request.form['password'] == 'student3' and request.form['username'] == 'student3'):
        session['account'] = request.form['username']
        session['logged_in'] = True
        return rootStudent()
    else :
        return render_template('loginerror.html')


@app.route('/login', methods=['GET'])
def login():
   return render_template('login.html')


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

@app.route("/predict", methods=['POST'])
def predict():
    acc = request.form['acc']
    feedback1 = request.form['feedback1']
    feedback1 = normalize_text(feedback1)
    sentiment1 = infer(modelA, tokenizer, feedback1)
    topic1 = infer(modelB, tokenizer, feedback1)
    time1 = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")
    write_to_csv_departments(time1,sentiment1[0],topic1[0],feedback1,acc)

    feedback2 = request.form['feedback2']
    if feedback2 != '':
        feedback2 = normalize_text(feedback2)
        sentiment2 = infer(modelA, tokenizer, feedback2)
        topic2 = infer(modelB, tokenizer, feedback2)
        time2 = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")
        write_to_csv_departments(time2,sentiment2[0],topic2[0],feedback2,acc)

    feedback3 = request.form['feedback3']
    if feedback3 !='':
        feedback3 = normalize_text(feedback3)
        sentiment3 = infer(modelA, tokenizer, feedback3)
        topic3 = infer(modelB, tokenizer, feedback3)
        time3 = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")
        write_to_csv_departments(time3,sentiment3[0],topic3[0],feedback3,acc)

    feedback4 = request.form['feedback4']
    if feedback4 !='':
        feedback4 = normalize_text(feedback4)
        sentiment4 = infer(modelA, tokenizer, feedback4)
        topic4 = infer(modelB, tokenizer, feedback4)
        time4 = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")
        write_to_csv_departments(time4,sentiment4[0],topic4[0],feedback4,acc)
    return render_template('thankyoupage.html')


@app.route('/admin')
def root():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        total_feedbacks, total_positive_feedbacks, total_negative_feedbacks, total_neutral_feedbacks, li, month = get_counts()
        tp,tn,tneu,cp,cn,cneu,lfp,lfn,lfneu,ecp,ecn,ecneu = li
        neg1,neu1,pos1,neg2,neu2,pos2,neg3,neu3,pos3,neg4,neu4,pos4,neg5,neu5,pos5,neg6,neu6,pos6,neg7,neu7,pos7,neg8,neu8,pos8,neg9,neu9,pos9,neg10,neu10,pos10,neg11,neu11,pos11,neg12,neu12,pos12 = month
        return render_template('admin.html',tf = total_feedbacks,tpf = total_positive_feedbacks,tnegf = total_negative_feedbacks, tneuf= total_neutral_feedbacks,
                               tp=tp,tn=tn,tneu=tneu,cp=cp,cn=cn,cneu=cneu,
                               lfp=lfp,lfn=lfn,lfneu=lfneu,ecp=ecp,
                               ecn=ecn,ecneu=ecneu, neg1 = neg1,neu1=neu1,pos1=pos1,neg2=neg2,neu2=neu2,pos2=pos2,neg3=neg3,neu3=neu3,pos3=pos3,neg4=neg4,neu4=neu4,pos4=pos4,neg5=neg5,neu5=neu5,pos5=pos5,neg6=neg6,neu6=neu6,pos6=pos6,neg7=neg7,neu7=neu7,pos7=pos7,neg8=neg8,neu8=neu8,pos8=pos8,neg9=neg9,neu9=neu9,pos9=pos9,neg10=neg10,neu10=neu10,pos10=pos10,neg11=neg11,neu11=neu11,pos11=pos11,neg12=neg12,neu12=neu12,pos12=pos12,
                               )

@app.route('/student')
def rootStudent():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        account = session['account']
        return render_template('index.html', acc = account)




@app.route("/display")
def display():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        df = pd.read_csv('dataset/database.csv')
        return render_template('feedbacks.html', tables=[df.to_html(classes='data', header="true")])

app.secret_key = os.urandom(12)
app.run(port=5978, host='0.0.0.0', debug=True)


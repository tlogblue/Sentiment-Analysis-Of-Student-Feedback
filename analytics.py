import csv
from MySQLdb import Timestamp
import pandas as pd
import numpy as np



def write_to_csv_departments(time,sentiment,topic,feedback,acc):

    with open('dataset/database.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for header in reader:
            break
    with open('dataset/database.csv', "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        dict = {'Timestamp': time, 'sentiment':sentiment,'topic':topic,'feedback':feedback, 'account':acc}
        writer.writerow(dict)



def get_counts():
    path = 'dataset/database.csv'
    df = pd.read_csv(path)
    index = df.index
    no_of_students = len(index)
    total_feedbacks = len(index)
    teaching_negative_count = ((df['topic']==0)&(df['sentiment']==0)).sum()
    #print(teaching_negative_count)
    teaching_neutral_count = ((df['topic']==0)&(df['sentiment']==1)).sum()
    #print(teaching_neutral_count)
    teaching_positive_count = ((df['topic']==0)&(df['sentiment']==2)).sum()
    #print(teaching_positive_count)
    #df1 = df.groupby('coursecontentscore').count()[['coursecontent']]
    coursecontent_negative_count = ((df['topic']==1)&(df['sentiment']==0)).sum()
    coursecontent_neutral_count = ((df['topic']==1)&(df['sentiment']==1)).sum()
    coursecontent_positive_count = ((df['topic']==1)&(df['sentiment']==2)).sum()

    #df1 = df.groupby('facilitiesscore').count()[['facilities']]
    libraryfacilities_negative_count = ((df['topic']==2)&(df['sentiment']==0)).sum()
    libraryfacilities_neutral_count = ((df['topic']==2)&(df['sentiment']==1)).sum()
    libraryfacilities_positive_count = ((df['topic']==2)&(df['sentiment']==2)).sum()

    #df1 = df.groupby('otherscore').count()[['other']]
    extracurricular_negative_count = ((df['topic']==3)&(df['sentiment']==0)).sum()
    extracurricular_neutral_count = ((df['topic']==3)&(df['sentiment']==1)).sum()
    extracurricular_positive_count = ((df['topic']==3)&(df['sentiment']==2)).sum()

    total_positive_feedbacks = teaching_positive_count + coursecontent_positive_count + libraryfacilities_positive_count + extracurricular_positive_count
    total_neutral_feedbacks = teaching_neutral_count + coursecontent_neutral_count + libraryfacilities_neutral_count + extracurricular_neutral_count
    total_negative_feedbacks = teaching_negative_count + coursecontent_negative_count + libraryfacilities_negative_count + extracurricular_negative_count
    
    neg1 = neg2 = neg3 = neg4 = neg5 = neg6 =neg7 =neg8 =neg9=neg10=neg11=neg12=0
    neu1 = neu2 = neu3 = neu4 = neu5 = neu6 =neu7 =neu8 =neu9=neu10=neu11=neu12=0
    pos1 = pos2 = pos3 = pos4 = pos5 = pos6 =pos7 =pos8 =pos9=pos10=pos11=pos12=0
    size = df['sentiment'].size
    for i in range (size):
        if df['Timestamp'][i][0:2] == "01":
            if df['sentiment'][i] == 0:
                neg1 = neg1 + 1
            elif df['sentiment'][i] == 1:
                neu1 = neu1 + 1
            else:
                pos1 = pos1 + 1
        elif df['Timestamp'][i][0:2] == "02":
            if df['sentiment'][i] == 0:
                neg2 = neg2 + 1
            elif df['sentiment'][i] == 1:
                neu2 = neu2 + 1
            else:
                pos2 = pos2 + 1
        elif df['Timestamp'][i][0:2] == "03":
            if df['sentiment'][i] == 0:
                neg3 = neg3 + 1
            elif df['sentiment'][i] == 1:
                neu3 = neu3 + 1
            else:
                pos3 = pos3 + 1
        elif df['Timestamp'][i][0:2] == "04":
            if df['sentiment'][i] == 0:
                neg4 = neg4 + 1
            elif df['sentiment'][i] == 1:
                neu4 = neu4 + 1
            else:
                pos4 = pos4 + 1
        elif df['Timestamp'][i][0:2] == "05":
            if df['sentiment'][i] == 0:
                neg5 = neg5 + 1
            elif df['sentiment'][i] == 1:
                neu5 = neu5 + 1
            else:
                pos5 = pos5 + 1
        elif df['Timestamp'][i][0:2] == "06":
            if df['sentiment'][i] == 0:
                neg6 = neg6 + 1
            elif df['sentiment'][i] == 1:
                neu6 = neu6 + 1
            else:
                pos6 = pos6 + 1
        elif df['Timestamp'][i][0:2] == "07":
            if df['sentiment'][i] == 0:
                neg7 = neg7 + 1
            elif df['sentiment'][i] == 1:
                neu7 = neu7 + 1
            else:
                pos7 = pos7 + 1
        elif df['Timestamp'][i][0:2] == "08":
            if df['sentiment'][i] == 0:
                neg8 = neg8 + 1
            elif df['sentiment'][i] == 1:
                neu8 = neu8 + 1
            else:
                pos8 = pos8 + 1
        elif df['Timestamp'][i][0:2] == "09":
            if df['sentiment'][i] == 0:
                neg9 = neg9 + 1
            elif df['sentiment'][i] == 1:
                neu9 = neu9 + 1
            else:
                pos9 = pos9 + 1
        elif df['Timestamp'][i][0:2] == "10":
            if df['sentiment'][i] == 0:
                neg10 = neg10 + 1
            elif df['sentiment'][i] == 1:
                neu10 = neu10 + 1
            else:
                pos10 = pos10 + 1
        elif df['Timestamp'][i][0:2] == "11":
            if df['sentiment'][i] == 0:
                neg11 = neg11 + 1
            elif df['sentiment'][i] == 1:
                neu11 = neu11 + 1
            else:
                pos11 = pos11 + 1
        else:
            if df['sentiment'][i] == 0:
                neg12 = neg12 + 1
            elif df['sentiment'][i] == 1:
                neu12 = neu12 + 1
            else:
                pos12 = pos12 + 1

    li = [teaching_positive_count,teaching_negative_count,teaching_neutral_count,
          coursecontent_positive_count,coursecontent_negative_count,coursecontent_neutral_count,
          libraryfacilities_positive_count,libraryfacilities_negative_count,libraryfacilities_neutral_count,
          extracurricular_positive_count,extracurricular_negative_count,extracurricular_neutral_count]
    month = [neg1,neu1,pos1,neg2,neu2,pos2,neg3,neu3,pos3,neg4,neu4,pos4,neg5,neu5,pos5,neg6,neu6,pos6,neg7,neu7,pos7,neg8,neu8,pos8,neg9,neu9,pos9,neg10,neu10,pos10,neg11,neu11,pos11,neg12,neu12,pos12]
    return no_of_students,\
           int(round((total_positive_feedbacks / total_feedbacks) * 100)),\
           int(round((total_negative_feedbacks / total_feedbacks )* 100)),\
           int(round((total_neutral_feedbacks / total_feedbacks )* 100)),\
            li,\
            month

def get_tables():
    df= pd.read_csv('dataset/database.csv')
    df = df.tail(5)
    return [df.to_html(classes='data')]

def get_titles():
    df = pd.read_csv('dataset/database.csv')
    return df.columns.values
get_counts()
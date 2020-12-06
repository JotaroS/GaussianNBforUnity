using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
public class MyGNBTest : MonoBehaviour
{
    // Start is called before the first frame update
    List<List<float>> train_data = new List<List<float>>();
    List<List<float>> test_data = new List<List<float>>();
    List<string> train_label = new List<string>();
    List<string> test_label = new List<string>();

    GaussianNB clf = new GaussianNB();

    public Dictionary<string, float> pred = new Dictionary<string, float>();
    void Start()
    {
        train_data = LoadData("Assets/train_states.txt");
        train_label = LoadLabel("Assets/train_labels.txt");
        test_data = LoadData("Assets/test_states.txt");
        test_label = LoadLabel("Assets/test_labels.txt");

        clf.Train(train_data,train_label);
        pred = clf.Predict(test_data[0]);

        foreach(var key in pred.Keys){
            Debug.Log(pred[key]);
        }
        
    }

    List<List<float>> LoadData(string path){
        List<List<float>> ret = new List<List<float>>();
        StreamReader sr = new StreamReader(path);
        while(sr.Peek() >= 0){
            string s = sr.ReadLine();
            string[] ss = s.Split(',');
            List<float> tmp = new List<float>();
            foreach(var s_ in ss){
                tmp.Add(Convert.ToSingle(s_));
            }
            ret.Add(tmp);
        }
        sr.Close();
        return ret;
    }
    List<string> LoadLabel(string path){
        List<string> ret = new List<string>();
        StreamReader sr = new StreamReader(path);
        while(sr.Peek() >= 0){
            string s = sr.ReadLine();
            ret.Add(s);
        }
        sr.Close();
        return ret;
    }
}

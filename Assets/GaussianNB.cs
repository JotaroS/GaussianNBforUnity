using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.Assertions;
using UnityEngine.UIElements;

public class GaussianNB
{
    Dictionary<string, float> p_class_ = new Dictionary<string, float>();
    Dictionary<string, List<List<float>>> f_stats_ = new  Dictionary<string, List<List<float>>>();
    List <string> labels_list_ = new List<string>();

    int features_count_;

    public GaussianNB(){

    }
    ~GaussianNB(){

    }
    public void Train(List <List<float>> data, List<string> labels){
        Dictionary <string, List<List<float>>> lfm = new Dictionary <string, List<List<float>>>();
        Dictionary <string, int> class_count = new Dictionary<string, int>();
        int train_size = labels.Count();

        labels_list_ = labels.Distinct().ToList();
        features_count_ = data[0].Count();

        foreach(var label in labels_list_){
            class_count[label]=0;
            List <float> arr= new float[features_count_].ToList<float>();//new List<float>(new float[features_count_]);
            List<List<float>> tmp = new List<List<float>>();
            tmp.Add(new float[features_count_].ToList<float>());
            tmp.Add(new float[features_count_].ToList<float>());
            tmp.Add(new float[features_count_].ToList<float>());
            f_stats_.Add(label,tmp); 
        }
        foreach(var label in labels_list_){
            lfm.Add(label, new List<List<float>>());
        }

        for(int i=0; i < train_size; i++){
            lfm[labels[i]].Add(data[i]);
            class_count[labels[i]] += 1;
            for(int j=0; j < features_count_; j++){
                f_stats_[labels[i]][0][j] += data[i][j]; // E(x at label i)
            }
        }

        foreach(var label in labels_list_){
            for(int j=0; j < features_count_; j++){
                f_stats_[label][0][j] /= class_count[label];
            }
            p_class_[label] = class_count[label] * 1.0f / train_size;
        }

        foreach(var label in labels_list_){
            for(int j=0; j < features_count_; j++){
                for(int i=0; i < lfm[label].Count(); i++){
                    f_stats_[label][1][j] += Mathf.Pow(lfm[label][i][j] - f_stats_[label][0][j], 2.0f); 
                }
                f_stats_[label][1][j] /= class_count[label]; //biased variance!
                Debug.Log(f_stats_[label][1][j]);
                f_stats_[label][2][j] = 1.0f / Mathf.Sqrt(2.0f * Mathf.PI * f_stats_[label][1][j]);
            }
        }
    }

    public Dictionary<string, float> Predict(List<float> vec)
    {
        Assert.IsTrue(features_count_ == vec.Count());

        Dictionary<string, float> ret = new Dictionary<string, float>();
        
        foreach(var label in labels_list_)
        {
            ret[label] = p_class_[label];
            for(int j = 0; j < features_count_; j++)
            {
                ret[label] *= f_stats_[label][2][j] * Mathf.Exp(-Mathf.Pow(vec[j] - f_stats_[label][0][j], 2) / (2 * f_stats_[label][1][j]));
            }
        }
        float norm = 0;
        foreach(var label in labels_list_){
            norm += ret[label];
        }
        Assert.IsTrue(norm != 0);
        foreach(var label in labels_list_){
            ret[label] /= norm;
        }
        return ret;
    }
    
};

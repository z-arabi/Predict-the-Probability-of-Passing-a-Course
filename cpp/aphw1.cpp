#include "aphw1.h"
#include <fstream>
#include <string>
#include <bits/stdc++.h> 
#include <cmath>
#include <math.h> 

std::vector<std::vector<double>> getData(const char* filename, bool add_bias)
{
    std::ifstream myFile(filename);
   
    // std::cout << filename << std::endl;

    std::vector<std::vector<double>> dataList;
    
    std::vector<std::string> tokens;
    std::vector<double> token;
    std::string intermediate;
    std::string outString{};
    
    while (getline(myFile,outString))
    {
        // std::cout << outString << std::endl;
        std::stringstream check1(outString); 
        while(getline(check1, intermediate, '\n')) 
            tokens.push_back(intermediate); 
        
        // for(int i = 0; i < tokens.size(); i++) 
        //     std::cout << tokens[i] << "**" << std::endl;
    }
    myFile.close();

    for (size_t i=0 ; i<tokens.size();i++)
    {
        std::stringstream check2(tokens[i]);
        while(getline(check2,intermediate,','))
            token.push_back(stod(intermediate));
        if (add_bias)
            token.insert(token.begin(),1);
        dataList.push_back(token);
        token = {};
    }
    // std::cout << dataList.size() << tokens.size() << " "<< std::endl;
    // std::cout << dataList[25][4] << " "<< std::endl;

    return dataList;
}

void displayDataset(std::vector <std::vector <double>> dataList, bool has_bias)
{
    if(has_bias)
    {
        std::cout << std::left << std::setfill(' ') << std::setw(15) << "No" 
                  << std::left << std::setfill(' ') << std::setw(15) << "Bias" 
                  << std::left << std::setfill(' ') << std::setw(15) << "Class"
                  << std::left << std::setfill(' ') << std::setw(15) << "TA"
                  << std::left << std::setfill(' ') << std::setw(15) << "Coding"
                  << std::left << std::setfill(' ') << std::setw(15) << "Studing"
                  << std::left << std::setfill(' ') << std::setw(15) << "Background"
                  << std::left << std::setfill(' ') << std::setw(15) << "Talent"
                  << "Passed"<< std::endl;
        std::cout << std::string(127,'*') << std::endl;
        
        for(int i = 0; i < static_cast<int>(dataList.size()); i++)
        {
            std::cout <<  std::left << std::setw(15) << i+1 ;
            for (int j = 0; j < static_cast<int>(dataList[i].size()); j++)
            {
                if (j == static_cast<int>(dataList[i].size()) -1 )
                    std::cout << std::left << dataList[i][j] << std::endl;
                else
                    std::cout << std::left << std::setw(15) << dataList[i][j];
            }                
        }        
    } else
    {
        std::cout << std::left << std::setfill(' ') << std::setw(15) << "No" 
                  << std::left << std::setfill(' ') << std::setw(15) << "Class"
                  << std::left << std::setfill(' ') << std::setw(15) << "TA"
                  << std::left << std::setfill(' ') << std::setw(15) << "Coding"
                  << std::left << std::setfill(' ') << std::setw(15) << "Studing"
                  << std::left << std::setfill(' ') << std::setw(15) << "Background"
                  << std::left << std::setfill(' ') << std::setw(15) << "Talent"
                  << "Passed"<< std::endl;
        std::cout << std::string(112,'*') << std::endl;
                  
        for(int i = 0; i < static_cast<int>(dataList.size()); i++)
        {
            std::cout <<  std::left << std::setw(15) << i+1 ;
            for (int j = 0; j < static_cast<int>(dataList[i].size()); j++)
            {
                if (j == static_cast<int>(dataList[i].size()) -1 )
                    std::cout << std::left << dataList[i][j] << std::endl;
                else
                    std::cout << std::left << std::setw(15) << dataList[i][j];
            }                
        }
    }
}

double h(std::vector<double> features, std::vector<double> w)
{
    double z{0};

    for(size_t i{0}; i<w.size() ; i++)
    {
        z = z + features[i]*w[i];
    }
    double h = 1/(1+exp(-z));
    // std::cout << z << " " << h << std::endl;

    return h;
}

double J(std::vector<std::vector<double>> data, std::vector<size_t> indices, std::vector<double> w)
{
    double H{0};
    double loss{0};
    double Sum{0};
    
    for (size_t j=0 ; j< indices.size() ; j++)
    {
        H = h(data[indices[j]],w);
        // std::cout << H << std::endl ;

        if (data[indices[j]][data[0].size()-1] == 1)
            loss = -log(H);
        else if(data[indices[j]][data[0].size()-1] == 0)
            loss = -log(1-H);
        else
        {
            std::cout << "Error" << std::endl;
        }        
        Sum += loss;
    }
    // std::cout << Sum << " " << indices.size() << " " << Sum/(indices.size()) << std::endl ;
    return Sum/(indices.size());
}

std::vector<double> fitOneEpoch(std::vector<std::vector<double>> data, std::vector<double> w0, double lr, size_t batch_size)
{
    
    std::vector<double> w = w0;
    int n = data.size() / batch_size;
    int m = data.size() % batch_size;

    for (int b{0}; b <= n ; b++)
    {
        for (size_t j{0}; j < w0.size(); j++)
        {
            for (size_t i{0}; i < batch_size; i++)
            {
                if( i + b*batch_size < data.size() )
                {
                    if (b == n){
                        // std::cout << i+b*batch_size <<std::endl;
                        w0[j] += (lr / m) * ( (data[i + b*batch_size][7] - h(data[i + b*batch_size] , w) ) *  data[i + b*batch_size][j] );
                    }else{
                        // std::cout << i+b*batch_size <<std::endl;
                        w0[j] += (lr / batch_size) * ( (data[i + b*batch_size][7] - h(data[i + b*batch_size] , w) ) *  data[i + b*batch_size][j] );
                    }
                }
            }
        }
        w = w0;
        // for (size_t i=0; i<w.size() ; i++)
        // {
        //     std::cout << b*8 << w[i] << "*";
        // }
        // std::cout<<std::endl;   
    }
    return w;
}

std::vector<double> fit(std::vector<std::vector<double>> data, std::vector<double> w0, double lr, size_t epochs, size_t batch_size, bool verbose)
{
    std::vector<double> w = w0;
    std::vector<size_t> ind;
    double sum{0};
    int n = data.size() / batch_size;

    for (size_t e=0 ; e<epochs ; e++)
    {
        w = fitOneEpoch(data,w,lr,batch_size);

        sum = 0;        
        if ( e == 0 || e == epochs-1 || verbose)
        {
            for (int b{0}; b <= n ; b++)
            {
                ind = {};         
                for (size_t i{0}; i < batch_size; i++)
                {
                    if( i + b*batch_size < data.size() )
                    {
                        ind.push_back(i+b*batch_size);
                    }
                }
                sum += J(data,ind,w);

                // for (auto in : ind)
                //     std::cout << in <<"*";
                // std::cout << "=>" << sum << "=>" << J(data,ind,w) <<std::endl;
            }            
            std::cout << "Epoch" << e << ":  " << sum/(static_cast<int>(data.size()/batch_size)+1) << std::endl;
        }           
    }
    return w;
}

std::vector<double> predict(std::vector<std::vector<double>> data, std::vector<double> w, bool verbose)
{
    double d{0};
    std::vector<double> result;

    for (size_t i{0} ; i<data.size() ; i++)
    {
        d = h(data[i],w);
        // std::cout << i << " " << d << std::endl;
    
        result.push_back(d);
    }

    // for (size_t i{0}; i<result.size() ; i++)
    //     std::cout << result[i] << std::endl;

    if (verbose)
    {
        std::cout << std::left << std::setfill(' ') << std::setw(20) << "Student Number" 
                  << std::left << std::setfill(' ') << std::setw(20) << "Result"
                  << "Pass/Fail" << std::endl;
        std::cout << std::string(50,'*') << std::endl;

        for (size_t i{0} ; i<data.size() ; i++)
        {
            std::cout << std::left << std::setfill(' ') << std::setw(20) << i+1
                      << std::left << std::setfill(' ') << std::setw(20) << result[i];
            if (result[i] > 0.5)
                std::cout << std::left << "Pass" << std::endl;
            else if (result[i] < 0.5)
                std::cout << std::left << "Fail" << std::endl;
        }
    }
    return result;
}
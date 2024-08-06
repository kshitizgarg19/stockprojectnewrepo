#include<iostream>
using namespace std;
bool search (int arr[],int target,int i,int size){
    if(i==size){
        return false;
    }
    if(arr[i]==target){
        return true;
    }
    return search(arr,target,i+1,size);
}
int main(){
    int arr[]={1,2,3};
    int i=0;
    bool ans =search(arr,2,i,2);
    if(ans==1) cout<<"true";
    else cout<<"false";
}
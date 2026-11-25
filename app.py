import streamlit as st
import torch
from PIL import Image
from src.model import load_model, features, scores
st.title('Part-Aware Few-Shot Recognition')
@st.cache_resource
def setup(): return load_model('cpu')
n=st.slider('Number of classes',2,5,5); k=st.radio('Shots',[1,5]); groups=[]; names=[]
for i in range(n):
    names.append(st.text_input(f'Class {i+1} name',f'Class {i+1}'))
    groups.append(st.file_uploader(f'Support images {i+1}',type=['jpg','jpeg','png'],accept_multiple_files=True,key=str(i)))
q=st.file_uploader('Query image',type=['jpg','jpeg','png'])
if st.button('Predict'):
    if not q or any(len(g)!=k for g in groups): st.error('Provide exactly the selected number of support images per class and one query.')
    else:
        model,t=setup()
        def encode(fs): return features(model,torch.stack([t(Image.open(f).convert('RGB')) for f in fs]))
        pairs=[encode(g) for g in groups]; qg,ql=encode([q]); s=(scores(torch.stack([p[0] for p in pairs]),qg)+scores(torch.stack([p[1] for p in pairs]),ql))/2
        st.image(q); st.success(names[int(s.argmax())]); st.write({name:float(v) for name,v in zip(names,s[0])}); st.caption('Cosine similarity scores, not calibrated probabilities.')

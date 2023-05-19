import os

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import transforms

from diploma import ResNetModel


def get_image_path(img):
    # Create a directory and save the uploaded image.
    file_path = f"images/{img.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(img.getbuffer())
    return file_path


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.resize((224, 224))
        return img.convert('RGB')


abr = ['BLA',
       'EBO',
       'EOS',
       'LYT',
       'MON',
       'MYB',
       'NGB',
       'PEB',
       'PLM',
       'RAND']

classes = ['Blast',
           'Erythroblast',
           'Eosinophil',
           'Lymphocyte',
           'Monocyte',
           'Myelocyte',
           'Band neutrophil',
           'Proerythroblast',
           'Plasma cell',
           'Random image']

class_to_idx = {classes[i]: i for i in range(len(classes))}
idx_to_class = {i: classes[i] for i in range(len(classes))}
idx_to_abr = {i: abr[i] for i in range(len(abr))}

detailed_information = {
    'Blast': 'Elevated levels of blasts are associated with a higher risk of progression of myelodysplastic disorder '
             'to AML (Acute myeloid leukemia)',
    'Erythroblast': 'Erythroblasts detected in the blood of patients may indicate the progression of a severe form of '
                    'anemia',
    'Eosinophil': 'Parasitic diseases and allergic reactions to drugs are among the most common causes of elevated '
                  'eosinophils',
    'Lymphocyte': 'A high level of lymphocytes can indicate a serious disease, for example: viral infections, '
                  'including measles, mumps and mononucleosis, adenovirus, hepatitis.',
    'Monocyte': 'Monocyte dysfunction can indicate a variety of serious diseases, such as sepsis and cancer',
    'Myelocyte': 'An increase in the number of myelocytes can be caused by leukemia',
    'Band neutrophil': 'An increased level of banded neutrophils is observed in the presence of infections in the body.',
    'Proerythroblast': 'A high level of proerythroblasts may indicate DBA (Diamond Blackfan Anemia)',
    'Plasma cell': 'Disorders of plasma cells can be signs of bone disease or kidney failure.',
    'Random image': 'Please, check input image'}

data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.RandomAdjustSharpness(sharpness_factor=5),
                                      transforms.RandomAutocontrast(),
                                      transforms.ToTensor(),
                                      ])


def answer(max_prob, max_prob_indices):
    key = idx_to_class.get(max_prob_indices[0].item())
    abrv = idx_to_abr.get(max_prob_indices[0].item())
    st.write("I guess, this is {} [{}] with probability {:.2f}%".format(key, abrv, max_prob[0]))
    st.caption("Details:")
    st.write(detailed_information.get(key))

    if max_prob[0] < 90:
        st.info('There are other classes with high probability:', icon="ℹ️")
        i = 1
        while (i < len(max_prob)) and (max_prob[i] > 5):
            st.write("{} [{}] with probability {:.2f}%".format(idx_to_class.get(max_prob_indices[i].item()),
                                                               idx_to_abr.get(max_prob_indices[i].item()), max_prob[i]))
            i += 1


st.set_page_config(page_title="Classification app", page_icon='icon.jpg')
st.title('Cytology data classification')

uploadFile = st.file_uploader(label="Upload image for classification", type=['jpg', 'png'])
if uploadFile is not None:
    img = get_image_path(uploadFile)

    st.caption("Your image:")
    st.image(img)

    img = pil_loader(img)
    img = data_transforms(img)

    model = ResNetModel(new_model=False)
    prediction = model.predict(img)
    max_prob, max_prob_indices = torch.topk(prediction, 3)

    st.caption("My prediction:")
    answer(max_prob[0], max_prob_indices[0])

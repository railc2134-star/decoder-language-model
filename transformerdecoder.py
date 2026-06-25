import torch
import torch.nn as nn
import csv
import re
vocab= {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
X=[]
Y=[]
with open("englishdataset.csv","r",newline='',encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        line = row.get("text").lower()
        X.append(row.get("text"))
        for letter in line:
            if letter not in vocab:
                vocab[letter] = len(vocab)

def char_to_word_to_sentance(text,vocab) :
    text=text.lower()
    sentance_id=[]
    for word in text:
        for letter in word:
            sentance_id.append(vocab.get(letter))
    return sentance_id
def fixing(ids):
    table=ids
    for i in range(len(table)):
        if len(table[i]) >=64:
            table[i]=table[i][:64]
        else:
            loss=64-len(table[i])
            table[i]=table[i]+[0]*loss
    return table
X_first=[]
for sentance in X:
    X_first.append(char_to_word_to_sentance(sentance,vocab))
X_first= X_first[0]
X_secand=[X_first[i:i+64]for i in range(0 ,len(X_first),64)]
Y=[X_first[i+1:i+65]for i in range(0,len(X_first),64)]
X_secand=fixing(X_secand)
Y=fixing(Y)
X_secand=torch.tensor(X_secand)
Y=torch.tensor(Y).long( )
class Decoder(nn.Module):
    def __init__(self, embad_size=128,head_size=4,vocab_size=len(vocab),ff_output=256):
        super().__init__()
        self.embaddings=nn.Embedding(vocab_size,embad_size)
        self.head=nn.MultiheadAttention(embad_size,head_size,batch_first=True)
        self.gps=nn.Parameter(torch.randn((1,64,embad_size)))
        self.agent=nn.Linear(embad_size,ff_output)
        self.dropout=nn.Dropout(0.3)
        self.boss=nn.Linear(ff_output,embad_size)
        self.boss2=nn.Linear(embad_size,vocab_size)
        self.norm1=nn.LayerNorm(embad_size)
        self.norm2=nn.LayerNorm(embad_size)
    def forward(self,x):
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).bool()
        x=self.embaddings(x)
        x=x+self.gps[:, :seq_len, :]
        original=x
        x,_=self.head(x,x,x,attn_mask=mask)
        x=self.norm1(x+original)
        linear=self.dropout(torch.relu(self.agent(x)))
        linear=self.boss(linear)
        x=self.norm2(linear+x)
        x=self.boss2(x)
        return x
pm=5
if pm==5 :
    model = Decoder(vocab_size=len(vocab))
    model.load_state_dict(torch.load('decoder.pth'))
    model.eval()
else:
    model=Decoder()
    edit=torch.optim.Adam(model.parameters(),lr=0.001)
    creation=nn.CrossEntropyLoss()
    for epoch in range(750):
        edit.zero_grad()
        outpute=model(X_secand)
        outpute = outpute.permute(0, 2, 1)
        loss=creation(outpute,Y)
        loss.backward()
        edit.step()
        if epoch %25==0:
         print(f"epoch = {epoch}|| loss = {loss}")
    print(f"last loss{loss}")
    torch.save(model.state_dict(),"decoder.pth")
    print(f"brain saved succesfuly")
def generation(model,input):
    ids = char_to_word_to_sentance(input, vocab)
    ids=torch.tensor(ids).unsqueeze(0)
    gen=[]
    id_to_char = {v: k for k, v in vocab.items()}
    for _ in range(500):
        outpute=model(ids)
        last=outpute[0,-1,:].argmax().item()
        if last==vocab["<EOS>"]:
            break
        gen.append(last)
        ids = torch.cat([ids, torch.tensor([[last]])], dim=1)
        if ids.shape[1] > 64:
            ids = ids[:, -64:]
    return "".join([id_to_char[i] for i in gen])
print(generation(model, "i"))
---
layout: page
title: Sheldon-bot
description:  Fine-tuning Microsoft's DialoGPT to create Sheldon-bot
img:
importance: 4
category: fun
toc:
  sidebar: left
---

I loved watching the big bang theory as a child.
Using my NLP and LLM knowledge, I decided to build my own Sheldon-bot I can chat with.

For source code, go to my [Github](https://www.github.com/hyeniii/sheldon-bot)

<div class="col-sm-8 mt-3 mt-md-0" style="float: left; margin-right: 20px;" >
    {% include figure.html path="https://static1.srcdn.com/wordpress/wp-content/uploads/2019/08/Sheldon-Cooper.v1.jpg?q=50&fit=crop&w=943&dpr=1.5" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

 ## Background

DialoGPT, a large-scale, pre-trained dialogue response generation model, serves as a perfect starting point for creating conversational agents. My objective was to customize this model to mimic Sheldon's unique conversational style, characterized by his idiosyncratic humor and intellect.

## Preparing the dataset:

I found the bigbang theory script on [Kaggle](https://www.kaggle.com/datasets/mitramir5/the-big-bang-theory-series-transcript). I convert this dataset in a way that every response row will contain 7 previous responses as a context. 

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reponse</th>
      <th>context</th>
      <th>context/0</th>
      <th>context/1</th>
      <th>context/2</th>
      <th>context/3</th>
      <th>context/4</th>
      <th>context/5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I think this is the place.</td>
      <td>If you have to ask, maybe you shouldn’t be here.</td>
      <td>Yes. Um, is this the High IQ sperm bank?</td>
      <td>Can I help you?</td>
      <td>One across is Aegean, eight down is Nabakov, ...</td>
      <td>Hang on.</td>
      <td>Excuse me?</td>
      <td>There’s no point, I just think it’s a good id...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Leonard, I don’t think I can do this.</td>
      <td>Oh, take your time. I’ll just finish my cross...</td>
      <td>Thank-you. We’ll be right back.</td>
      <td>Fill these out.</td>
      <td>I think this is the place.</td>
      <td>If you have to ask, maybe you shouldn’t be here.</td>
      <td>Yes. Um, is this the High IQ sperm bank?</td>
      <td>Can I help you?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No. We are committing genetic fraud. There’s ...</td>
      <td>What, are you kidding? You’re a semi-pro.</td>
      <td>Leonard, I don’t think I can do this.</td>
      <td>Oh, take your time. I’ll just finish my cross...</td>
      <td>Thank-you. We’ll be right back.</td>
      <td>Fill these out.</td>
      <td>I think this is the place.</td>
      <td>If you have to ask, maybe you shouldn’t be here.</td>
    </tr>
  </tbody>
</table>
</div>

## Initial Interaction with DialoGPT
```

User: Hi Sheldon
DialoGPT: I'm not sure if you're joking or not.

User: Where is Leonard
DialoGPT: I'm not sure if you're joking or not.

User: Love is in the air
DialoGPT: I love Leonard

User: why do you love Leonard
DialoGPT: I love Leonard

User: Do you know Sheldon Cooper
DialoGPT: I love Sheldon
```
DialoGPT seems to be confused for sure! Let me try to fine tune this GPT!!

## After fine tuning 
```
User: Hi Sheldon
Sheldonbot:  Hello.

User: Where is lenoard
Sheldonbot:  It’s Leonard’ s office.

User: What are your thoughts on the big bang theory
Sheldonbot:  I don’t know. It‘s a little cliche, but it has some really great implications for quantum cosmology.

User: love is in the air
Sheldonbot:  No, it is not.

User: what is in the air
Sheldonbot: !!!ERN!!
```
Sounds more like Sheldon to me!! haha
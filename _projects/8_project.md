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

##
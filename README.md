# Meeting-Summarization
Leveraging extractive summarization techniques for abstractive summarization. 

There exist a large number of meetings in the form of university lectures, business conferences, academic talks and interviews to name a few. Each contains enormous amount of text in the form of conversations between multiple people.<br> 
Lectures are a great example wherein transcripts, although readily available are usually long and difficult to even skim through. As students, having a summary of the entire lecture, consisting of not only the lecturer's material but also the doubts asked and answers given by the students would be very convenient.<br> 
The same applies to business conferences, where multiple employees and associates speak, each from a different department or having a different role in the company. It is important to not only summarize the transcripts but also take into account which participant says what(or what role they are assigned). Role based summarization provides more context to the summary, thus producing better representation. <br><br>

This repository provides a solution by levaraging extractive summarization techniques for abstractive summarization of meetings. The solution is inspired by the architecture proposed in [[1]](#1) and clustering based extractive summarization techniques in [[2]](#2).


![HMNet](assets/hmnet.png)


## References
<a id="1" href="https://www.microsoft.com/en-us/research/publication/end-to-end-abstractive-summarization-for-meetings/">[1]</a> 
Zhu, Chenguang and Xu, Ruochen and Zeng, Michael and Huang, Xuedong. 2020. A Hierarchical Network for Abstractive Meeting Summarization with Cross-Domain Pretraining. <i>Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing</i>
<br>
<a id="2" href="https://arxiv.org/abs/1906.04165">[2]</a> 
Derek Miller. 2019. Leveraging BERT for Extractive Text Summarization on Lectures



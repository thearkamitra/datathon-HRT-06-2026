You are a financial analyst in charge of multiple portfolios and must take decisions based on the headlines that you have access to.

Read the data folder and the README.md present in that folder to understand the structure of the data.

In src, add a file that is responsible for understanding the sentiment of the headlines and contain metadata that is imporant for those.

The headlines should have a class of its own that contain all of the headlines.
Inside that class, each headline should have information about where it has appeared and how the previous sentiment about that has changed.
The headline should identify which company it pertains to or maybe it is not relevant at all.
Additionally, it should have a method to predict what the present headline gives of as a sentiment, whether it is supposed to cause a buy or sell with reason.
If there is additional information about that company (only looking back), it should use that as additional context in a systematic manner.

To get the prediction, it can either choose the GEMINI_API or the ollama local models that are present to provide the output.
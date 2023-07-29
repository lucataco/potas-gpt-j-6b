import banana_dev as client
from io import BytesIO
import base64
import time

# Localhost test
my_model = client.Client(
    api_key="",
    model_key="",
    url="http://localhost:8000",
)

inputs = {
    "prompt": '''Given a review from Amazon's food products, the task is to generate a short summary of the given review in the input.
Input: I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than most.
Output: Good Quality Dog Food

Input: Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was an error or if the vendor intended to represent the product as 'Jumbo'.
Output: Not as Advertised

Input: My toddler loves this game to a point where he asks for it. That's a big thing for me. Secondly, no glitching unlike one of their competitors (PlayShifu). Any tech I don’t have to reach out to support for help is a good tech for me. I even enjoy some of the games and activities in this. Overall, this is a product that shows that the developers took their time and made sure people would not be asking for refund. I’ve become bias regarding this product and honestly I look forward to buying more of this company’s stuff. Please keep up the great work.
Output:''',
}

inputs2 = {
    "prompt": '''Please answer the following question:

Question: What is the capital of Canada?
Answer: Ottawa

Question: What is the currency of Switzerland?
Answer: Swiss franc

Question: In which country is Wisconsin located?
Answer:''',
}

inputs3 = {
    "prompt": '''Label the tweets as either "positive", "negative", "mixed", or "neutral":

Tweet: I can say that there isn't anything I would change.
Label: positive

Tweet: I'm not sure about this.
Label: neutral

Tweet: I liked some parts but I didn't like other parts.
Label: mixed

Tweet: I think the background image could have been better.
Label: negative

Tweet: I really like it.
Label:''',
}

# Call your model's inference endpoint on Banana.
# If you have set up your Potassium app with a
# non-default endpoint, change the first
# method argument ("/")to specify a
# different route.
t1 = time.time()
result, meta = my_model.call("/", inputs3)
t2 = time.time()

output = result["output"]
print(output)
print("Time to run: ", t2 - t1)
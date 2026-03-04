FEW_SHOT_EXAMPLES = {
    "sst2": [
        {
            "text": "been discovered , indulged in and rejected as boring before i see this piece of crap again ",
            "analysis": "The text contains strongly negative phrases such as 'rejected as boring' and 'piece of crap'. These words explicitly convey a highly unfavorable, dismissive, and angry opinion. Therefore, the sentiment is negative.",
            "label": "negative"
        },
        {
            "text": "at every opportunity to do something clever ",
            "analysis": "The phrase 'do something clever' carries a favorable and constructive connotation. It implies praise for ingenuity and positive action. Therefore, the sentiment is positive.",
            "label": "positive"
        }
    ],
    
    "semeval": [
        {
            "text": "This chap seems to be a bit of an over sexed out going extrovert ... Must be his overly masculine voice and demeanor.",
            "analysis": "The text uses ellipsis and hyperbolic descriptions like 'over sexed out going extrovert' and 'overly masculine'. The phrase 'Must be' creates a mocking tone, strongly suggesting that the speaker actually means the exact opposite of what is written to make fun of the subject. This incongruity indicates irony.",
            "label": "ironic"
        },
        {
            "text": "Long Gold Layered Pipe Necklace Set $20|Leave email for invoice",
            "analysis": "This is a straightforward commercial advertisement selling a necklace set for $20. The language is purely transactional and literal, with no hidden meaning, exaggeration, or contradiction. Therefore, it is non-ironic.",
            "label": "non-ironic"
        }
    ],

    "isarcasm": [
        {
            "text": "'Margaret Thatcher' is trending... has 2016 taken another one?",
            "analysis": "Margaret Thatcher passed away in 2013. The year 2016 was notable for the deaths of many celebrities. The author is pretending to believe she might have died again in 2016 just because her name is trending. This feigned ignorance and dark humor signify intended sarcasm.",
            "label": "ironic"
        },
        {
            "text": "I’m ready for Lockdown #2. I need to make better bread.",
            "analysis": "The author is simply stating a personal goal—wanting to make better bread—during a second lockdown. The statement aligns perfectly with the literal meaning of the words and expresses a genuine intention without any sarcastic undertones.",
            "label": "non-ironic"
        }
    ],

    "ag_news": [
        {
            "text": "Oil-price boom a boon for producers Not everyone was complaining when the price of a barrel of oil flirted with the \\$50 mark on global markets earlier this summer. For Venezuelan President Hugo Chavez, it meant more money to ",
            "analysis": "The text explicitly discusses 'Oil-price', 'producers', 'global markets', and monetary values ('$50 mark'). These are core economic and financial terms indicating a discussion about commerce, trading, and market trends. Thus, it is a Business news item.",
            "label": "Business"
        },
        {
            "text": "Dell goes wireless with printers Company adds new level of convenience to home and office printing with wireless printer adapter and multifunction inkjet product.\\&lt;br /&gt;\\  Photo: Dell All-In-One Printer 962 \\",
            "analysis": "The text mentions 'Dell', 'wireless printer adapter', and 'multifunction inkjet product'. These terms clearly describe computing hardware, electronics, and technological innovations. Hence, it belongs to the Sci/Tech category.",
            "label": "Sci/Tech"
        },
        {
            "text": "Beltre, Dodgers belt Mets Adrian Beltre went 5 for 5 with his major league-leading 42d home run, and the Los Angeles Dodgers got a pair of big pinch hits yesterday in a 4-2 victory over the Mets in New York.",
            "analysis": "The presence of terms like 'Dodgers', 'Mets', 'home run', and 'major league' firmly grounds this text in the context of baseball. This is unmistakably a description of a competitive Sports event.",
            "label": "Sports"
        },
        {
            "text": "Bush cites opening for peace, leadership WASHINGTON -- President Bush offered a powerful incentive to Palestinians yesterday to rally behind a moderate leadership, signaling the promise of US reengagement and a new opening for peace following Palestinian leader Yasser Arafat's death.",
            "analysis": "The text covers international diplomacy, mentioning 'President Bush', 'Palestinians', 'leadership', and a 'new opening for peace'. Because it deals with geopolitical events and international relations, it falls under the World news category.",
            "label": "World"
        }
    ],

    "trec": [
        {
            "text": "What is Mikhail Gorbachev 's middle initial ?",
            "analysis": "The question explicitly asks for a 'middle initial', which is a single letter used to represent a name. This syntactic structure perfectly aligns with seeking the full form or character of an Abbreviation.",
            "label": "Abbreviation"
        },
        {
            "text": "Where did the energy for the Big Bang come from ?",
            "analysis": "The question asks for an explanation or reason ('Where did the energy... come from') regarding a complex cosmic phenomenon. It requires a descriptive explanatory answer rather than a simple name or number. Thus, it seeks a Description.",
            "label": "Description"
        },
        {
            "text": "What color were their horses ?",
            "analysis": "The question asks for a 'color', which is a specific attribute or entity. It does not ask for a person, location, or numerical value, making Entity the most appropriate broad category.",
            "label": "Entity"
        },
        {
            "text": "What was the name of Randy Craft 's lawyer ?",
            "analysis": "The question asks for the 'name of... lawyer'. A lawyer is a professional role held by a person. Therefore, the expected answer is a Human being.",
            "label": "Human"
        },
        {
            "text": "Where is the Kalahari desert ?",
            "analysis": "The question uses the interrogative word 'Where' and asks for the position of a geographical feature (a desert). This explicitly requires a Location as the answer.",
            "label": "Location"
        },
        {
            "text": "When did beethoven die ?",
            "analysis": "The question starts with 'When', which specifically requires a date, year, or time as the answer. Dates and times fall under the broader category of numerical values, so the answer type is Number.",
            "label": "Number"
        }
    ]
}
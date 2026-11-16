from reactagenthlt.preprocessing import gsm8k, hotpotqa

if __name__ == "__main__":
    qa = hotpotqa(setting="distractor", train=False)
    print(qa.head())
    print(qa.info())
    print(qa.describe())

    print("=" * 50)

    gsm = gsm8k(setting="main", train=False, reasoning=False)
    print(gsm.head())
    print(gsm.info())
    print(gsm.describe())

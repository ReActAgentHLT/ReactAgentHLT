from reactagenthlt.preprocessing import load_gsm8k, load_hotpotqa

if __name__ == "__main__":
    qa = load_hotpotqa(setting="distractor", train=False)
    print(qa.head())
    print(qa.info())
    print(qa.describe())

    print("=" * 50)

    gsm = load_gsm8k(setting="main", train=False, reasoning=False)
    print(gsm.head())
    print(gsm.info())
    print(gsm.describe())

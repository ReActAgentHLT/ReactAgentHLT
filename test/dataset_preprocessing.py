from reactagenthlt.preprocessing import gsm8k, hotpotqa

if __name__ == "__main__":
    qa = hotpotqa(setting="distractor", train=False)
    print(qa)

    print("=" * 50)

    gsm = gsm8k(setting="main", train=False, reasoning=False)
    print(gsm)

"""Programmatic arithmetic-expression and word-problem generator."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
from dataclasses import dataclass
import hashlib
import random

KINDS=("single","chain2","chain4","poly2","poly3","word")
ASK_TRAIN=("What is {e}?","Calculate {e}.","Compute {e}.","Solve {e}.","Evaluate {e}.")
ASK_HELDOUT=("Determine the value of {e}.","Work out {e}.","Find the result when you evaluate {e}.")
WRAP_TRAIN=(("Question: {q}\nAnswer:"," "),("Problem: {q}\nSolution:"," "),("{q}\nThe answer is"," "),("{q}\n","Answer: "))
WRAP_HELDOUT=(("Exercise: {q}\nShow your work:"," "),("Please answer this: {q}\nResponse:"," "))
OBJECTS=("apples","books","marbles","stickers","pencils","coins","cards","flowers")
CONSONANTS="bcdfghjklmnprstvwz"
VOWELS="aeiou"

@dataclass
class Example:
    kind:str;prompt:str;answer:str;gold:int

@dataclass
class Node:
    text:str;value:int;steps:list[str];precedence:int=9

def choose(rng,x):return x[rng.randrange(len(x))]
def random_name(rng):
    length=rng.randint(3,7);use_consonant=bool(rng.getrandbits(1));letters=[]
    for _ in range(length):
        letters.append(choose(rng,CONSONANTS if use_consonant else VOWELS));use_consonant=not use_consonant
    return "".join(letters).capitalize()
def partition(key):return "val" if int(hashlib.sha1(key.encode()).hexdigest()[:8],16)%20==0 else "train"
def shown(n,parent_prec):return f"({n.text})" if n.precedence<parent_prec else n.text

def combine(rng,left,right,op):
    prec=2 if op in ("*","/") else 1;symbol=choose(rng,("x","×","*")) if op=="*" else op
    value={"+":left.value+right.value,"-":left.value-right.value,"*":left.value*right.value,"/":left.value//right.value}[op]
    text=f"{shown(left,prec)} {symbol} {shown(right,prec+int(op in ('-','/')))}"
    return Node(text,value,left.steps+right.steps+[f"{left.value} {symbol} {right.value} = {value}"],prec)

def random_expression(rng,operators):
    value=rng.randint(0,99);node=Node(str(value),value,[])
    for _ in range(operators):
        op=choose(rng,("+","-","*","/"));rv=rng.randint(1,30);right=Node(str(rv),rv,[])
        if op=="/":
            # Pick a divisor of the current result, preserving the preceding subtree.
            divisors=[d for d in range(1,31) if node.value%d==0]
            rv=choose(rng,divisors);right=Node(str(rv),rv,[])
        node=combine(rng,node,right,op)
        if abs(node.value)>100000:break
    return node

def polynomial(rng):
    degree=choose(rng,(2,3));x=rng.randint(-8,8);coeff=[rng.randint(-12,12) for _ in range(degree+1)]
    if coeff[0]==0:coeff[0]=rng.choice((-1,1))*rng.randint(1,12)
    terms=[];values=[]
    for i,a in enumerate(coeff):
        power=degree-i;value=a*x**power;values.append(value);terms.append(f"({a})x^{power}" if power>1 else f"({a})x" if power==1 else f"({a})")
    total=sum(values);expr=" + ".join(terms)+f" at x = {x}";steps=[f"Substitute x = {x}."]
    for i,(a,value) in enumerate(zip(coeff,values)):
        power=degree-i;steps.append(f"Term {i+1}: {a} x {x}^{power} = {value}.")
    # Reduce the signed terms pairwise.  The previous single four-term sum was
    # the dominant source of otherwise-correct polynomial failures.
    running=values[0]
    for value in values[1:]:
        new_total=running+value
        steps.append(f"Running sum: {running} + ({value}) = {new_total}.")
        running=new_total
    return Node(expr,total,steps)

def word_problem(rng):
    name,other,obj=random_name(rng),random_name(rng),choose(rng,OBJECTS)
    kind=rng.randrange(10)
    if kind==0:
        start=rng.randint(2,999);used=rng.randint(1,start)
        verb=choose(rng,("eats","gives away","loses","sells","uses"));gold=start-used
        q=f"{name} has {start} {obj} and {verb} {used} of them. How many {obj} remain?"
        return q,Node(f"{start} - {used}",gold,[])
    if kind==1:
        start,more=rng.randint(0,999),rng.randint(1,999);verb=choose(rng,("buys","receives","finds","collects","is given"));gold=start+more
        q=f"{name} has {start} {obj} and {verb} {more} more. How many {obj} does {name} have now?"
        return q,Node(f"{start} + {more}",gold,[])
    if kind==2:
        a,b=rng.randint(0,999),rng.randint(0,999);gold=a+b
        q=f"{name} has {a} {obj} and {other} has {b}. How many {obj} do they have altogether?"
        return q,Node(f"{a} + {b}",gold,[])
    if kind==3:
        groups,each=rng.randint(1,50),rng.randint(1,50);gold=groups*each
        q=f"There are {groups} boxes with {each} {obj} in each box. How many {obj} are there altogether?"
        return q,Node(f"{groups} x {each}",gold,[])
    if kind==4:
        people,each=rng.randint(1,50),rng.randint(0,50);total=people*each
        q=f"{total} {obj} are shared equally among {people} people. How many does each person receive?"
        return q,Node(f"{total} / {people}",each,[])
    if kind==5:
        start=rng.randint(1,500);add=rng.randint(1,300);remove=rng.randint(1,start+add);mid=start+add;gold=mid-remove
        q=f"{name} starts with {start} {obj}, receives {add}, and then gives away {remove}. How many remain?"
        return q,Node(f"{start} + {add} - {remove}",gold,[f"After receiving more: {start} + {add} = {mid}",f"After giving some away: {mid} - {remove} = {gold}"])
    if kind==6:
        groups,each,extra=rng.randint(1,30),rng.randint(1,30),rng.randint(0,50);mid=groups*each;gold=mid+extra
        q=f"{name} fills {groups} bags with {each} {obj} in each and then adds {extra} loose {obj}. How many {obj} are there?"
        return q,Node(f"{groups} x {each} + {extra}",gold,[f"The bags contain {groups} x {each} = {mid}",f"Then add {extra}: {mid} + {extra} = {gold}"])
    if kind==7:
        groups,red,blue=rng.randint(1,20),rng.randint(1,30),rng.randint(1,30);per=red+blue;gold=groups*per
        q=f"Each of {groups} baskets contains {red} red and {blue} blue {obj}. How many {obj} are in all the baskets?"
        return q,Node(f"({red} + {blue}) x {groups}",gold,[f"Each basket has {red} + {blue} = {per}",f"Across {groups} baskets: {per} x {groups} = {gold}"])
    if kind==8:
        # Explicit multi-addend abstraction, reduced one addition at a time.
        names=[name,other,random_name(rng),random_name(rng)]
        amounts=[rng.randint(0,300) for _ in names]
        running=amounts[0];steps=[]
        for amount in amounts[1:]:
            new_total=running+amount
            steps.append(f"Running total: {running} + {amount} = {new_total}")
            running=new_total
        q=", ".join(f"{n} has {a}" for n,a in zip(names[:-1],amounts[:-1]))+f", and {names[-1]} has {amounts[-1]} {obj}. How many {obj} do they have altogether?"
        return q,Node(" + ".join(map(str,amounts)),running,steps)
    start=rng.randint(20,500);changes=[rng.randint(1,100) for _ in range(3)]
    # Keep the story non-negative while teaching a mixed running-state update.
    changes[1]=min(changes[1],start+changes[0]);changes[2]=min(changes[2],start+changes[0]-changes[1])
    first=start+changes[0];second=first-changes[1];gold=second-changes[2]
    q=f"{name} has {start} {obj}, gets {changes[0]} more, gives away {changes[1]}, and then uses {changes[2]}. How many remain?"
    steps=[f"After getting more: {start} + {changes[0]} = {first}",f"After giving some away: {first} - {changes[1]} = {second}",f"After using some: {second} - {changes[2]} = {gold}"]
    return q,Node(f"{start} + {changes[0]} - {changes[1]} - {changes[2]}",gold,steps)

def generate(rng,split="train",forced_kind=None,heldout_template=False):
    while True:
        word=(forced_kind=="word") if forced_kind else rng.random()<.25
        poly=(forced_kind in ("poly2","poly3")) if forced_kind else (not word and rng.random()<.25)
        if word:q,node=word_problem(rng)
        else:
            if poly:
                node=polynomial(rng)
                while forced_kind and ((forced_kind=="poly2") != ("x^2" in node.text and "x^3" not in node.text)):
                    node=polynomial(rng)
            else:
                operators={"single":1,"chain2":2,"chain4":4}.get(forced_kind,rng.randint(1,5))
                node=random_expression(rng,operators)
            q=choose(rng,ASK_HELDOUT if heldout_template else ASK_TRAIN).format(e=node.text)
        key=("word|" if word else "expr|")+node.text
        if partition(key)==split:break
    wrappers=WRAP_HELDOUT if heldout_template else WRAP_TRAIN
    if len(node.steps)>1 and not heldout_template:
        wrappers=(WRAP_TRAIN[0],WRAP_TRAIN[1],WRAP_TRAIN[3])
    wrap,prefix=choose(rng,wrappers)
    body=(f"{node.text} = {node.value}" if rng.random()<.5 else str(node.value)) if len(node.steps)<=1 else ". ".join(s.rstrip(".") for s in node.steps)+f". Therefore, the answer is {node.value}."
    return Example(forced_kind or "mixed",wrap.format(q=q),prefix+body+"\n\n",node.value)

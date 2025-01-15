# Reinforcement Learning in Logic Synthesis

**Logic synthesis** is an integral part of **electronic design automation (EDA)**, involving optimizing a Boolean network derived from a hardware description language such as Verilog or VHDL, in regards to the number of nodes (area of the circuit) or the depth of the graph (circuit's delay). 

Once HDL is compiled, unoptimized circuits are represented in the form of and-inverter graphs, and a number of algorithms exist for the purpose of optimizing them by removing redundant nodes or rewriting to reduce graph depth. The order in which these algorithms are called upon greatly influences the final quallity of results, but finding optimal synthesis recipes is computationally impractical. Previous works attempted to train RL agents to approximate optimal synthesis recipes, and achieved so with results varying in validity and efficiency. Often times, these experiments trained on larger AIGs and infered on smaller ones, failing to provide models that successfully generalize their experience to larger scale circuits. 

One of the pitfalls of former attempts was reflected in poor state representations presented to the agent, which failed to recognize important distinctions between similar AIGs. This project aims to design a state representation that provides information relevant for logic synthesis.

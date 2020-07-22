# PyHTM
 This is a Python 3 implementation of Hierarchical Temporal Memory (HTM). It's fairly sparse (pun intended) as of right now, but it's functional. I built it mainly as a way to learn how HTM works--learn by doing, right? I thought it might be helpful for other newcomers to HTM as well.
 
 For those of you who aren't familiar with HTM, it is a model of intelligence proposed by Jeff Hawkins, co-founder of the organization Numenta that does open-source research on HTM. The model, based on the biology of the brain's neocortex, proposes that intelligence emerges from systems that are able to perform generalized learning and make predictions based on that learning. The model assumes that most processing happening in the brain is the same basic algorithm, repeated over and over with cross-linked connections.
 
 As I understand it, there are four basic components to HTM:
 
  1) The sparse-distributed-array (SDR). All data flowing through the HTM is in the format of a binary array with a low concentration of nonzero elements (~4%, for example). 
  2) Encoders. These take non-SDR inputs and turn them into SDRs. The key to learning is making it so that each distinct input has a unique SDR, but also allowing inputs that are      semantically similar to have some overlap in their nonzero bits. This will let the HTM know that those SDRs have some relationship, I.E. numbers that are unequal, but close.
  3) Spatial Poolers (SPs). These process SDRs and produce output SDRs. They learn to recognize data. Each bit in the output is represented by one 'minicolumn', which compares the      input SDR to its own internal connections list. If enough of the input bits to which it is connected are on, the minicolumn activates. Minicolumns can learn new connections        and forget old ones. Each minicolumn tends to learn strong connections to input bits which tend to be active together. The minicolumns all work together to produce an              internally consistent representation of the input. 
  4) Temporal Memory (TM). A TM can be paired with an SP to observe the temporal patterns in the SP's activity. The TM contains 'cells' which are similar to minicolumns. Each          minicolumn has a matching set of cells, and when the minicolumn activates at least one of its cells must activate as well. The pattern of WHICH cells activate allow the TM to      identify not only a specific input, but its place in a larger pattern or sequence of inputs.
 
 In summary: the idea behind the HTM algorithm is to encode all data--of whatever kind--as SDRs and use SPs and TMs to learn the spatiotemporal patterns of the data. The 'Hierarchical' in HTM is a nod to the idea of using multiple layers of SPs and TMs. Numenta's theory is that the brain creates intelligence by interfacing this system with sensory inputs (E.G. think of your eyes as visual encoders, your ears as audiostream encoders, etc.) and sending signals to the muscle control nerves as the output.
  
Shout-out to psdyer, whose guide gave me good ideas when I was starting this project: https://github.com/psdyer/NuPIC-Algorithm-API-Example/blob/master/algorithm_api_example.ipynb

In addition to the main library file (PyHTM.py), I have included several shorter scripts that endeavor to demonstrate how to use the system, along with figures that show what their outputs should look like.

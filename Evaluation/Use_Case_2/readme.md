This directory contains two subdirectories with the following structure:  

\1-Reference_Model\  
    create_reference_model.py - Python script used to create the reference model from the event log  
    Event log for reference model.xes - Event log containing the reference model (in a single trace)  
    reference_model.pdf - the prescribed reference model as a PDF file  
    reference_model.png - the prescribed reference model as a PNG file  
    reference_model.pnml - the prescribed reference model as Petri Net Markup Language file, as required by \2-Conformance_Checking\conformance_checking.py  

\2-Conformance_Checking\  
    conformance_checking.py - Python script used to recognize a deviation in a trace  
    nonconforming_event_log.xes - event log of a single trace that does not conform to the prescribed process model  
    urgent_note_procurement.pptx - PowerPoint file that contains instructions on when to act differently, thus incorporating a new process variant

\3-LLM-Responses\  
    1-Prompt.txt - The Prompt to the LLM for comparison. Used twice; before and after vectorizing the above .pptx  
    2-Response_Without_Domain_Knowledge.txt - full length LLM response without access to the .pptx  
    3-Response_With_Vectorized_Domain_Knowledge.txt - full length LLM response with access to the .pptx  


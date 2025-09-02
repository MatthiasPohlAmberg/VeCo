This directory contains two subdirectories with the following structure:  
\1-Data\  
    1-voicemail_speakers_text - the text used by the speaker to create the voicemail  
    2-voicemail.mp3 - the audio file used in the use case  
    3-voicemail_transcription.txt - what OpenAI whisper understood from the .mp3 file (and what got embedded in the database) for comparison  

\2-Prompts\  
    1-Prompt - The Prompt to the LLM for comparison. Used twice; before and after vectorizing the above.mp3  
    2-Response_Without_Domain_Knowledge.txt - the full response of the LLM without access to the voicemail.  
    3-Response_With_Domain_Knowledge.txt - the full response of the LLM with access to the audio file.  
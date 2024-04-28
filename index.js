let recorder;
let audioChunks = [];

function startRecording() {
    audioChunks = []; 
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = e => {
                audioChunks.push(e.data);
            };
            recorder.start();
            document.getElementById("recordButton").disabled = true;
            document.getElementById("stopButton").disabled = false;
        })
        .catch(err => console.error(err));
}

function stopRecording() {
    recorder.stop();
    document.getElementById("recordButton").disabled = false;
    document.getElementById("stopButton").disabled = true;

    // Wait for the recorder to stop
    recorder.onstop = () => {
        // Combine the audio chunks into a single blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/*' });
        sendDataToServer(audioBlob);
    };
}

function sendDataToServer(audioBlob) {
    const formData = new FormData();
    formData.append("audio", audioBlob);
    console.log("Sending audio blob:", audioBlob);

    fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
       // alert(`Result: ${data.result}`);
    })
    .catch(err => console.error(err));
}

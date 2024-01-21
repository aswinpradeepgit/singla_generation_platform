let hoverCount = 0;

document.getElementById('yesButton').addEventListener('click', function() {
    document.getElementById('questionSection').style.display = 'none';
    document.getElementById('responseSection').style.display = 'block';
});

document.getElementById('noButton').addEventListener('mouseover', function(event) {
    hoverCount++;
    if (hoverCount < 4) {
        const maxX = window.innerWidth - event.target.offsetWidth;
        const maxY = window.innerHeight - event.target.offsetHeight;
        event.target.style.position = 'absolute';
        event.target.style.left = Math.random() * maxX + 'px';
        event.target.style.top = Math.random() * maxY + 'px';
    } else {
        event.target.style.opacity = '0';
        event.target.style.transition = 'opacity 2s';
        setTimeout(() => event.target.remove(), 2000); // Remove button after fade-out
    }
});

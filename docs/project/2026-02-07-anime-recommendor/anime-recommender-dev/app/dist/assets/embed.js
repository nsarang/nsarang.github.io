function o(e){document.body.classList.add("embedded"),new ResizeObserver(()=>{window.parent.postMessage({type:"resize",height:e.scrollHeight},"*")}).observe(e)}export{o as initEmbed};

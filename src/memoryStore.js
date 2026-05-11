(function () {
    function exportMemory() {
        let memoryObject = {};
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith('gomoku-') || key.startsWith('tensorflowjs_models/gomoku-adaptive-weights/')) {
                memoryObject[key] = localStorage.getItem(key);
            }
        }

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(memoryObject));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "omokbot_memory.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }

    function importMemory(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            try {
                const memoryObject = JSON.parse(e.target.result);
                let isValid = false;
                let importedKeys = 0;

                localStorage.clear();
                for (const key in memoryObject) {
                    if (key.startsWith('gomoku-') || key.startsWith('tensorflowjs_models/')) {
                        localStorage.setItem(key, memoryObject[key]);
                        isValid = true;
                        importedKeys++;
                    }
                }

                if (isValid) {
                    alert(`성공적으로 이어하기 데이터를 불러왔습니다! (${importedKeys}개의 세이브 파일)\n새로고침 됩니다.`);
                    location.reload();
                } else {
                    alert("적합한 오목봇 세이브 파일이 아닙니다.");
                }
            } catch (error) {
                alert("세이브 파일을 읽는 중 오류가 발생했습니다: " + error.message);
            }

            event.target.value = '';
        };
        reader.readAsText(file);
    }

    window.OmokMemoryStore = {
        exportMemory,
        importMemory
    };
})();

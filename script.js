$(document).ready(function(){
    var API_KEY = "AIzaSyAQq8GytdgxBMDeyguvudsakpo4NzAK3sQ"

    var video = ''


    $("#form").submit(function (event) {
        event.preventDefault()

        var emotion = $("#search").val()
        var search

        if(emotion == "sad"){
            search = "rap live stream"
        } else if (emotion == "angry"){
            search = "pop"
        } else {
            search = emotion
        }




        videoSearch(API_KEY,search,1)
    }) 

    function videoSearch(key,search, maxResults){
        $.get("https://www.googleapis.com/youtube/v3/search?part=snippet&key=" + key + "&q=" + search + "&type=video&maxResults=" + maxResults,
        function (data) {
            console.log(data)

            data.items.forEach(item => {
                video = `<iframe width="420" height="40" src="https://www.youtube.com/embed/${item.id.videoId}?&autoplay=1" frameborder="0"></iframe>`

                $("#videos").append(video)

            });
        })
    }

})

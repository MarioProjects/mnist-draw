

$(document).ready(function () {

    
    $( "#make_predict" ).click(function() {
        var canvas = document.getElementById('canvas_draw');
        var canvas_data = canvas.toDataURL();

        // Make prediction by calling api /predictModel
        $.ajax({
            type: 'POST',
            url: '/predictModel',
            data: canvas_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                data = JSON.parse(data);
                if (data.number!=undefined){
                    $("#number_info").text("Number: " + data.number)
                }else{
                    $("#number_info").text("No number")
                }
            },
        });
        
    });

    $("#clear_canvas").click(function(){
        var $canvas = $("canvas");
        var context = $canvas[0].getContext("2d");
        var canvas = document.getElementById('canvas_draw');
        context.clearRect(0, 0, canvas.width, canvas.height);
    });

});


var color = $(".selected").css("background-color");
var $canvas = $("canvas");
//Select the first, only canvas element. Select the actual HTML element using the array syntax [index], get the 2d context.
var context = $canvas[0].getContext("2d");
context.strokeStyle = color;
context.lineJoin = "round";
context.lineWidth = 34;
context.globalAlpha = 0.73;

var lastEvent;
var mouseDown = false;

//On mouse events on the canvas
$canvas.bind("touchstart mousedown", function(e) {
    lastEvent = e;
    mouseDown = true;

}).bind("touchmove mousemove", function(e) {
    e.preventDefault();
    e.stopPropagation();
    if (mouseDown) {
        //Draw lines
        context.beginPath();
        context.moveTo(lastEvent.offsetX, lastEvent.offsetY);
        context.lineTo(e.offsetX, e.offsetY);
        context.closePath();        
        context.stroke();
        lastEvent = e;
    }
}).bind("mouseup", function() {
    mouseDown = false;
}).bind("touchend mouseleave", function() {
    $canvas.mouseup();
});


$( document ).ready(function() {

    var relevant = [];
    var irrelevant = [];

    $(document).on("click", ".radio_button", function () {
        console.log($(this).attr('tag'), this.name);
        var tag = $(this).attr('tag');
        var name = this.name;
        if(tag == "relevant"){

            if(irrelevant.includes(name)){
                var index = irrelevant.indexOf(name);
                irrelevant.splice(index, index+1);
            }
            relevant.push(name);

        }
        else if(tag == "irrelevant"){

             if(relevant.includes(name)){
                 var index = relevant.indexOf(name);
                 relevant.splice(index, index+1);
            }
            irrelevant.push(name);
        }

   });

    $(document).on("click","#submit_feedback_button",function () {
        console.log("Clicked");
        var formData = new FormData();

        var name = $(this).attr('rel_type');
        var q = $(this).attr('q');
        var t = parseInt($(this).attr('t'));
        formData.append('relevant[]', JSON.stringify(relevant));
        formData.append('irrelevant[]', JSON.stringify(irrelevant));        
        formData.append('rel_type',JSON.stringify(name));
        formData.append('q',JSON.stringify(q))
        formData.append('t',JSON.stringify(t))
        
        $.ajax({
            type: 'POST',
            url: '/get_feedback',
            data: formData,
            processData: false,
            contentType: false,

            success: function(data){
                var result = $('<div />').append(data).find('#showResults').html();
                $('#showResults').html(result).show();
                $('.heading').show();
                $('#load').css('display','none');
                relevant = [];
                irrelevant = [];
            },
            beforeSend:function(){
                $('#load').css('display','block');
                $('.heading,#showResults').hide();
            },
            error: function (err_msg) {
                console.log("error ", err_msg);
                $('#load').hide();
            }
        });

    });

});